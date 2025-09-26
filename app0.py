
import io
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

# ------------------------------
# Config
# ------------------------------
RSEED = 42
st.set_page_config(page_title="Churn → Car Suggestions", layout="wide")
DEFAULT_CARS_XLSX = "auto_combined_updated.xlsx"

# =======================================================
# 1) Churn model helpers (same as before)
# =======================================================
def featurize(df: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    df = df.copy()
    df["BalancePositive"] = (df["Balance"] > 0).astype(int)
    df["Senior"] = (df["Age"] >= 60).astype(int)
    df["YoungAdult"] = ((df["Age"] >= 18) & (df["Age"] <= 30)).astype(int)
    df["Bal_Salary_Ratio"] = np.where(df["EstimatedSalary"] > 0, df["Balance"] / df["EstimatedSalary"], 0.0)
    df["CreditBucket"] = pd.cut(df["CreditScore"], [-np.inf, 500, 650, 750, np.inf],
                                labels=["very_low", "low", "mid", "high"]).astype(str)
    df["AgeBucket"] = pd.cut(df["Age"], [-np.inf, 25, 35, 45, 55, 65, np.inf],
                             labels=["<=25","26-35","36-45","46-55","56-65","65+"]).astype(str)
    df["TenureBucket"] = pd.cut(df["Tenure"], [-np.inf,1,3,5,7,9,np.inf],
                                labels=["<=1","2-3","4-5","6-7","8-9","10+"]).astype(str)
    cat_cols = ["Geography","Gender","CreditBucket","AgeBucket","TenureBucket"]
    num_cols = [c for c in df.columns if c not in cat_cols + ["Exited"]]
    return df, cat_cols, num_cols

def iqr_cap(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    df = df.copy()
    for c in num_cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        df[c] = np.clip(df[c], lower, upper)
    return df

def ohe_align(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: list):
    Xtr = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)
    Xte = pd.get_dummies(test_df,  columns=cat_cols, drop_first=True)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    return Xtr, Xte

def train_rf(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_split=2, min_samples_leaf=1,
        max_features="sqrt", bootstrap=True, class_weight="balanced",
        n_jobs=-1, random_state=RSEED
    )
    rf.fit(X_train, y_train)
    return rf

def summarize(y_true, y_pred, proba) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_true, proba)
    return {"Accuracy": round(acc,3), "Churn Precision": round(prec,3),
            "Churn Recall": round(rec,3), "Churn F1": round(f1,3), "ROC AUC": round(auc,3)}

@st.cache_data
def load_data():
    df_raw = pd.read_csv("data.csv")
    df = df_raw.drop(columns=["RowNumber","Surname","CustomerId"])
    return df_raw, df

@st.cache_data
def prepare_data(df):
    df_fe, cat_cols, num_cols = featurize(df)
    df_fe = iqr_cap(df_fe, num_cols)
    X = df_fe.drop(columns=["Exited"])
    y = df_fe["Exited"]
    return df_fe, X, y, cat_cols

@st.cache_resource
def fit_model(X, y, cat_cols):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RSEED, stratify=y)
    X_train_rf, X_test_rf = ohe_align(X_train, X_test, cat_cols)
    rf = train_rf(X_train_rf, y_train)
    proba_test = rf.predict_proba(X_test_rf)[:,1]
    pred_test  = (proba_test >= 0.50).astype(int)
    metrics = summarize(y_test, pred_test, proba_test)
    feature_cols = X_train_rf.columns.tolist()
    return rf, feature_cols, metrics

# =======================================================
# 2) Car catalog loader + suggestion logic
# =======================================================
def _first_present(columns: list[str], candidates: list[str]) -> Optional[str]:
    for cand in candidates:
        if cand in columns:
            return cand
    return None

def _coerce_price(series: pd.Series) -> pd.Series:
    if series.dtype.kind in {"i","u","f"}:
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.replace(r"[^0-9,\.]", "", regex=True)
    use_dot = cleaned.str.contains(r"\.")
    cleaned = np.where(use_dot, cleaned.str.replace(",", "", regex=False),
                       cleaned.str.replace(".", "", regex=False))
    cleaned = pd.Series(cleaned).str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")

@st.cache_data(show_spinner=False)
def load_cars_from_any(path_or_file) -> Tuple[pd.DataFrame, dict]:
    if hasattr(path_or_file, "read"):
        cars = pd.read_excel(path_or_file)
    else:
        src = str(path_or_file)
        cars = pd.read_excel(src) if src.lower().endswith((".xlsx",".xls")) else pd.read_csv(src)

    cars = cars.copy()
    cars.columns = [c.strip().lower() for c in cars.columns]
    cols = cars.columns.tolist()

    seg_col = _first_present(cols, ["segment","class","category","market_segment","kategori","segmento","seg"])
    if seg_col is None:
        st.error("Cars file must include a segment/class/category column."); st.stop()

    style_col = _first_present(
        cols, ["style","body","type","body_type","bodytype","bodystyle","body_style",
               "kasa","kasa tipi","kasa_tipi","gövde","govde","tip","tipi"]
    )
    if style_col is None:
        st.error("Cars file must include a style/body/type column."); st.stop()

    name_col = _first_present(cols, ["car_name","name","model","title","arac","araç","model_adi","model adı"])
    if name_col is None:
        maybe_brand = _first_present(cols, ["brand","make"])
        maybe_model = _first_present(cols, ["model","variant","trim"])
        if maybe_brand and maybe_model:
            cars["car_name"] = cars[maybe_brand].astype(str).str.strip().str.title() + " " + \
                               cars[maybe_model].astype(str).str.strip().str.title()
            name_col = "car_name"
        else:
            cars["car_name"] = cars.index.astype(str); name_col = "car_name"

    brand_col = _first_present(cols, ["brand","make"])
    if brand_col is None:
        cars["brand"] = cars[name_col].astype(str).str.strip().str.split().str[0].str.title()
        brand_col = "brand"
    else:
        cars["brand"] = cars[brand_col].astype(str).str.strip().str.title()

    price_col = _first_present(cols, ["price_numeric","price","msrp","list_price","fiyat","sale_price"])
    if price_col is not None:
        cars["price_numeric"] = _coerce_price(cars[price_col])

    seg_map = {"luxary":"Luxury","luxury":"Luxury","premium":"Luxury",
               "economy":"Economy","eko":"Economy","economic":"Economy",
               "mid":"Mid","middle":"Mid","medium":"Mid"}
    style_map = {"family":"Family","general":"General","sport":"Sport","compact":"Compact",
                 "sedan":"General","hatchback":"Compact","suv":"Family","crossover":"Family"}

    def norm_segment(x: str) -> str:
        x_clean = str(x).strip().lower()
        return seg_map.get(x_clean, x_clean.title())

    def tokenize_styles(x: str) -> list[str]:
        raw = str(x).strip().lower()
        parts = raw.replace("/", " ").replace(",", " ").replace("-", " ").split()
        norm = []
        for t in parts:
            t_map = style_map.get(t, t.title())
            if t_map not in norm:
                norm.append(t_map)
        return norm or [raw.title()]

    cars["segment"] = cars[seg_col].apply(norm_segment)
    cars["style"] = cars[style_col].astype(str).str.strip().str.title()
    cars["style_tokens"] = cars["style"].apply(tokenize_styles)
    cars["car_name"] = cars[name_col].astype(str).str.strip()

    mapping = {"segment": seg_col, "style": style_col, "car_name": name_col,
               "brand": brand_col, "price_numeric": price_col}
    return cars, mapping

def _age_bucket(age: int) -> str:
    return "<=25" if age<=25 else "26-35" if age<=35 else "36-45" if age<=45 else "46-55" if age<=55 else "56-65" if age<=65 else "65+"

def _wanted_styles(age_bucket: str) -> list[str]:
    if age_bucket in ["<=25","26-35"]:
        return ["Sport","Compact"]
    if age_bucket in ["46-55","56-65","65+"]:
        return ["Family","General"]
    return ["General","Compact","Family","Sport"]

def _segment_from_salary(salary: float) -> str:
    return "Economy" if salary < 50_000 else ("Mid" if salary < 150_000 else "Luxury")

def _candidate_pool(age: int, salary: float, cars: pd.DataFrame, exclude_brands: Optional[list[str]] = None) -> pd.DataFrame:
    need_segment = _segment_from_salary(float(salary))
    styles = _wanted_styles(_age_bucket(int(age)))

    def style_match(tokens: list[str], wanted: list[str]) -> bool:
        return any(t in wanted for t in tokens)

    pool = cars[(cars["segment"] == need_segment) & (cars["style_tokens"].apply(lambda t: style_match(t, styles)))]
    if exclude_brands and "brand" in pool.columns:
        pool = pool[~pool["brand"].isin([b.title() for b in exclude_brands])]

    if pool.empty:
        pool = cars[cars["segment"] == need_segment]
    if pool.empty:
        pool = cars[cars["style_tokens"].apply(lambda t: style_match(t, styles))]
    if pool.empty:
        pool = cars
    return pool

def suggest_car(age: int, estimated_salary: float, cars: pd.DataFrame, *,
                topk: int = 5, tie_eps: float = 0.05, per_user_seed: Optional[int] = None,
                diversify_noise: float = 0.02, exclude_brands: Optional[list[str]] = None,
                avoid_consecutive_same_brand: bool = True) -> Optional[pd.Series]:
    pool = _candidate_pool(age, estimated_salary, cars, exclude_brands)
    if pool.empty: return None

    if "price_numeric" in pool.columns and pool["price_numeric"].notna().any():
        budget = 0.6 * float(estimated_salary)
        priced = pool["price_numeric"].notna()
        base = pd.Series(index=pool.index, dtype=float)
        base.loc[priced] = -(pool.loc[priced,"price_numeric"] - budget).abs()
        base.loc[~priced] = -abs(budget) * 0.05
        score = base
    else:
        score = pd.Series(0.0, index=pool.index)

    if diversify_noise > 0:
        rng = np.random.default_rng(per_user_seed)
        score = score + rng.normal(0, diversify_noise, size=len(score))

    ranked = pool.assign(score=score).sort_values("score", ascending=False)
    best = ranked.iloc[0]["score"]
    near = ranked[ranked["score"] >= best - abs(best)*tie_eps]
    near = near.head(max(topk, len(near))) if not near.empty else ranked.head(max(1, topk))
    pick = near.sample(1, random_state=per_user_seed).iloc[0] if per_user_seed is not None else near.sample(1).iloc[0]
    if avoid_consecutive_same_brand and "brand" in near.columns:
        st.session_state["last_suggested_brand"] = pick.get("brand")
    return pick

# NEW: return 1–2 unique recommendations for a single customer
def suggest_cars_for_person(age: int, salary: float, cars: pd.DataFrame, *,
                            n_recs: int = 2, topk: int = 8, tie_eps: float = 0.05,
                            diversify_noise: float = 0.02, per_user_seed: Optional[int] = None,
                            exclude_brands: Optional[list[str]] = None) -> pd.DataFrame:
    n_recs = max(1, min(2, int(n_recs)))  # only 1 or 2
    pool = _candidate_pool(age, salary, cars, exclude_brands)
    if pool.empty:
        return pd.DataFrame()

    # score like above
    if "price_numeric" in pool.columns and pool["price_numeric"].notna().any():
        budget = 0.6 * float(salary)
        priced = pool["price_numeric"].notna()
        base = pd.Series(index=pool.index, dtype=float)
        base.loc[priced] = -(pool.loc[priced,"price_numeric"] - budget).abs()
        base.loc[~priced] = -abs(budget) * 0.05
        score = base
    else:
        score = pd.Series(0.0, index=pool.index)

    rng = np.random.default_rng(per_user_seed)
    if diversify_noise > 0:
        score = score + rng.normal(0, diversify_noise, size=len(score))

    ranked = pool.assign(score=score).sort_values("score", ascending=False)
    best = ranked.iloc[0]["score"]
    near = ranked[ranked["score"] >= best - abs(best)*tie_eps]
    near = near.head(max(topk, len(near))) if not near.empty else ranked.head(max(1, topk))

    # sample up to n unique rows (stable with seed)
    if len(near) <= n_recs:
        return near[["car_name","brand","segment","style","car_price"]].reset_index(drop=True)
    picks = near.sample(n=n_recs, random_state=per_user_seed, replace=False)
    return picks[["car_name","brand","segment","style","price_numeric"]].reset_index(drop=True)

# =======================================================
# 3) UI / flow
# =======================================================
st.title("Churn Prediction → Car Suggestions")

# Load + fit
df_raw, df_model = load_data()
df_fe, X_all, y_all, cat_cols = prepare_data(df_model)
rf, feature_cols, metrics = fit_model(X_all, y_all, cat_cols)

#st.subheader("Model (Random Forest) — quick summary")
#st.write(metrics)

with st.expander("Preview customer data (engineered)"):
    st.dataframe(df_fe.head(10))

# Churn threshold
st.subheader("Choose churn threshold")
th = st.slider("Decision threshold for P(churn)", 0.05, 0.95, 0.30, 0.01)

# Predict all rows
X_full = df_fe.drop(columns=["Exited"])
X_full_rf = pd.get_dummies(X_full, columns=cat_cols, drop_first=True).reindex(columns=feature_cols, fill_value=0)
proba_full = rf.predict_proba(X_full_rf)[:,1]
pred_full  = (proba_full >= th).astype(int)

st.subheader(f"Model summary (threshold = {th:.2f})")
if "Exited" in df_fe.columns:
    st.json(summarize(df_fe["Exited"], pred_full, proba_full))
else:
    st.info("Labels not available to compute metrics at current threshold.")

# Predictions DF
pred_df = df_fe.copy()
pred_df["ChurnProb"] = proba_full
pred_df["ChurnPred"] = pred_full
pred_df["CustomerId"] = df_raw["CustomerId"]

# ---------- Sidebar: catalog + controls ----------
with st.sidebar:
    st.subheader("Car catalog")
    uploaded = st.file_uploader("Upload Excel/CSV", type=["xlsx","xls","csv"])
    use_default = st.checkbox("Use default file", value=True, help=DEFAULT_CARS_XLSX)
    path_input = st.text_input("Or path", value=DEFAULT_CARS_XLSX if use_default else "")

    st.markdown("---")
    st.subheader("Suggestion controls")
    ui_topk  = st.slider("Top-K pool", 1, 20, 5, 1)
    ui_tie   = st.slider("Near-tie window (± of best)", 0.0, 0.25, 0.05, 0.01)
    ui_noise = st.slider("Diversify noise (σ)", 0.0, 0.1, 0.02, 0.005)

# Load cars
source = uploaded if uploaded is not None else (path_input if path_input.strip() else None)
if source is None:
    st.info("Upload/select a car catalog to generate recommendations.")
    st.stop()

cars, mapping = load_cars_from_any(source)
st.caption(
    f"Mapped columns — segment: `{mapping['segment']}`, style: `{mapping['style']}`, "
    f"name: `{mapping['car_name']}`, brand: `{mapping['brand'] or 'derived'}`, "
    f"price: `{mapping['price_numeric'] or 'none'}`"
)
with st.expander("Cars: catalog preview"):
    st.dataframe(cars.head(20))

# =======================================================
# NEW: Single-customer quick recommendations
# =======================================================
st.subheader("quick recommendations")

cc1, cc2, cc3, cc4 = st.columns([1,1,1,1])
with cc1:
    ui_age = st.number_input("Age", min_value=16, max_value=99, value=42, step=1)
with cc2:
    ui_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=5_000_000.0, value=120_000.0, step=1_000.0, format="%0.0f")
with cc3:
    n_recs = st.radio("How many?", [1,2], horizontal=True, index=1)
with cc4:
    # dynamic brand list
    brand_opts = sorted(cars["brand"].dropna().unique().tolist()) if "brand" in cars.columns else []
    ui_exclude = st.multiselect("Exclude brands", options=brand_opts, default=[])

if st.button("Recommend for this person", type="primary"):
    per_seed = (int(ui_age) * 10_000 + int(ui_salary)) % (2**32 - 1)
    recs = suggest_cars_for_person(
        age=int(ui_age),
        salary=float(ui_salary),
        cars=cars,
        n_recs=int(n_recs),
        topk=max(ui_topk, 5),
        tie_eps=ui_tie,
        diversify_noise=ui_noise,
        per_user_seed=per_seed,
        exclude_brands=ui_exclude,
    )
    if recs.empty:
        st.warning("No suitable car found.")
    else:
        st.success("Recommendation(s):")
        st.dataframe(recs.rename(columns={"price_numeric":"price"}), use_container_width=True)
        # small Excel download
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            recs.to_excel(w, index=False, sheet_name="recommendations")
        st.download_button("Download this recommendation (Excel)",
                           data=buf.getvalue(),
                           file_name="single_customer_recommendations.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =======================================================
# Batch: churn-risk customers
# =======================================================
st.subheader("Car suggestions for at-risk customers")
churners = pred_df[pred_df["ChurnPred"] == 1].copy()
if churners.empty:
    st.info("No churn-risk customers at this threshold.")
else:
    all_brands = sorted(cars["brand"].dropna().unique().tolist()) if "brand" in cars.columns else []
    ui_exclude_batch = st.multiselect("Exclude brands (batch)", options=all_brands, default=[],
                                      help="Applied to the churners list below.")

    def row_to_suggestion(row):
        seed = int(row["CustomerId"]) if "CustomerId" in row else None
        pick = suggest_car(
            age=int(row["Age"]),
            estimated_salary=float(row["EstimatedSalary"]),
            cars=cars,
            topk=ui_topk,
            tie_eps=ui_tie,
            per_user_seed=seed,
            diversify_noise=ui_noise,
            exclude_brands=ui_exclude_batch,
            avoid_consecutive_same_brand=False,
        )
        return None if pick is None else pick.get("car_name")

    churners["SuggestedCar"] = churners.apply(row_to_suggestion, axis=1)
    churners = churners[churners["SuggestedCar"].notna()]

    show_cols = ["CustomerId","Geography","Gender","Age","AgeBucket","EstimatedSalary","ChurnProb","SuggestedCar"]
    show_cols = [c for c in show_cols if c in churners.columns]
    st.dataframe(churners[show_cols].sort_values("ChurnProb", ascending=False).head(50), use_container_width=True)

    out = churners[show_cols].sort_values("ChurnProb", ascending=False).reset_index(drop=True)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        out.to_excel(writer, index=False, sheet_name="churn_car_suggestions")
    st.download_button(
        "Download suggestions (Excel)",
        data=buffer.getvalue(),
        file_name="churn_customers_with_car_suggestions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.caption("Tip: Lower the threshold to increase recall (catch more churners); raise it to increase precision.")
