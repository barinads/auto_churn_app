import io
import re
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score,
    precision_recall_fscore_support
)
from sklearn.ensemble import RandomForestClassifier

# ------------------------------
# Config
# ------------------------------
RSEED = 42
st.set_page_config(page_title="Churn → Car Suggestions", layout="wide")

# ------------------------------
# Feature Engineering (same for train & predict)
# ------------------------------
def featurize(df: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    """Return engineered df (copy), cat_cols, num_cols."""
    df = df.copy()

    # Flags & ratios
    df["BalancePositive"] = (df["Balance"] > 0).astype(int)
    df["Senior"] = (df["Age"] >= 60).astype(int)
    df["YoungAdult"] = ((df["Age"] >= 18) & (df["Age"] <= 30)).astype(int)
    df["Bal_Salary_Ratio"] = np.where(
        df["EstimatedSalary"] > 0, df["Balance"] / df["EstimatedSalary"], 0.0
    )

    # Buckets
    df["CreditBucket"] = pd.cut(
        df["CreditScore"],
        [-np.inf, 500, 650, 750, np.inf],
        labels=["very_low", "low", "mid", "high"]
    ).astype(str)

    df["AgeBucket"] = pd.cut(
        df["Age"],
        [-np.inf, 25, 35, 45, 55, 65, np.inf],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"]
    ).astype(str)

    df["TenureBucket"] = pd.cut(
        df["Tenure"],
        [-np.inf, 1, 3, 5, 7, 9, np.inf],
        labels=["<=1", "2-3", "4-5", "6-7", "8-9", "10+"]
    ).astype(str)

    cat_cols = ["Geography", "Gender", "CreditBucket", "AgeBucket", "TenureBucket"]
    num_cols = [c for c in df.columns if c not in cat_cols + ["Exited"]]
    return df, cat_cols, num_cols

def iqr_cap(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """Winsorize numerics by IQR caps."""
    df = df.copy()
    for c in num_cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[c] = np.clip(df[c], lower, upper)
    return df

def ohe_align(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: list):
    """One-hot encode + align columns."""
    Xtr = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)
    Xte = pd.get_dummies(test_df,  columns=cat_cols, drop_first=True)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    return Xtr, Xte

# ------------------------------
# Model helpers
# ------------------------------
def train_rf(X_train, y_train):
    """Train a solid RF quickly (fixed params to keep the app snappy)."""
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RSEED
    )
    rf.fit(X_train, y_train)
    return rf

def summarize(y_true, y_pred, proba) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, proba)
    return {
        "Accuracy": round(acc, 3),
        "Churn Precision": round(prec, 3),
        "Churn Recall": round(rec, 3),
        "Churn F1": round(f1, 3),
        "ROC AUC": round(auc, 3),
    }

# ------------------------------
# Car labeling (derive from price/brand/body_type) + strict suggestions
# ------------------------------
def _to_float_price(v):
    """Parse numeric price from strings like '$32,500', '€45.9k', etc."""
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).lower().replace('k','000')
    s = re.sub(r'[^0-9.\-]', '', s)
    try:
        return float(s)
    except:
        return np.nan

@st.cache_data
def load_and_label_cars(path: str,
                        price_bins=(0, 30_000, 70_000, float('inf')),
                        price_labels=("Economy", "Mid", "Luxury")) -> pd.DataFrame:
    """
    Load Excel, normalize columns, derive:
      - price_numeric
      - segment (Economy/Mid/Luxury) from price; fallback by brand tier
      - style (Sport/Family/Compact/General) from body_style/name
      - impute missing price_numeric via segment→brand median, segment median, overall median
    """
    cars = pd.read_excel(path)
    cars.columns = [c.lower() for c in cars.columns]

    # Identify likely columns
    price_col = next((c for c in ["car_price","price","msrp","listing_price","sale_price","retail_price"] if c in cars.columns), None)
    body_col  = next((c for c in ["body_style","body_type","body","vehicle_type","class","category"] if c in cars.columns), None)
    brand_col = next((c for c in ["brand","make","manufacturer","marque","company"] if c in cars.columns), None)
    name_col  = "car_name" if "car_name" in cars.columns else (next((c for c in ["name","model","title"] if c in cars.columns), None))

    if name_col is None:
        cars["car_name"] = cars.index.astype(str)
        name_col = "car_name"

    # 1) price_numeric
    if price_col:
        cars["price_numeric"] = cars[price_col].map(_to_float_price)
    else:
        cars["price_numeric"] = np.nan

    # 2) segment from price bins
    cars["segment"] = pd.cut(cars["price_numeric"], bins=price_bins, labels=price_labels, include_lowest=True)

    # 3) backfill segment by brand tier if missing
    if brand_col:
        brand = cars[brand_col].astype(str).str.strip().str.title()
        luxury_brands  = {"Audi","Bmw","Mercedes-Benz","Mercedes","Lexus","Infiniti","Acura","Volvo",
                          "Jaguar","Land Rover","Porsche","Maserati","Genesis","Cadillac","Lincoln",
                          "Alfa Romeo","Tesla","Bentley","Rolls-Royce","Aston Martin"}
        economy_brands = {"Toyota","Honda","Hyundai","Kia","Nissan","Renault","Peugeot","Citroen",
                          "Skoda","Seat","Dacia","Fiat","Opel","Vauxhall","Chevrolet","Ford","Mazda",
                          "Subaru","Suzuki","Mitsubishi","Volkswagen","Vw","Buick","Dodge","Chrysler",
                          "Ram","Pontiac","Saab","Holden"}
        def brand_to_segment(b):
            if b in luxury_brands:  return "Luxury"
            if b in economy_brands: return "Economy"
            return None
        cars["_brand_seg"] = brand.map(brand_to_segment)
        cars["segment"] = cars["segment"].astype(object).fillna(cars["_brand_seg"])

    # 4) style from body / name cues
    body_text = cars[body_col].astype(str).str.lower() if body_col else pd.Series("", index=cars.index)
    name_text = cars[name_col].astype(str).str.lower()

    sport_kw   = r"(?:^|\W)(sport|gt|gti|rs|sti|m |amg|type r|nismo|vrs|gr |srt|cupra|n line)(?:\W|$)"
    family_kw  = r"suv|crossover|mpv|minivan|van|wagon|estate"
    compact_kw = r"hatch|compact|subcompact|city"

    style = pd.Series("General", index=cars.index, dtype=object)
    # sporty by body/name badges
    style[name_text.str.contains(sport_kw, regex=True) | body_text.str.contains(r"coupe|roadster|cabrio|convertible")] = "Sport"
    # family by body
    style[body_text.str.contains(family_kw, regex=True)] = "Family"
    # compact by body
    style[body_text.str.contains(compact_kw, regex=True)] = "Compact"
    cars["style"] = style

    # 5) impute missing price_numeric for better ranking
    cars["segment"] = cars["segment"].astype(str).str.title()
    if brand_col:
        cars["brand_clean"] = cars[brand_col].astype(str).str.title()
    else:
        cars["brand_clean"] = "Unknown"

    cars["price_numeric"] = pd.to_numeric(cars["price_numeric"], errors="coerce")
    # by segment + brand
    cars["price_numeric"] = cars.groupby(["segment","brand_clean"])["price_numeric"]\
                                .transform(lambda s: s.fillna(s.median()))
    # by segment
    cars["price_numeric"] = cars.groupby("segment")["price_numeric"]\
                                .transform(lambda s: s.fillna(s.median()))
    # overall
    cars["price_numeric"] = cars["price_numeric"].fillna(cars["price_numeric"].median())

    # Keep essentials first
    lead = [c for c in ["car_name", "segment", "style", "price_numeric", body_col, brand_col] if c in cars.columns]
    cars = cars[lead + [c for c in cars.columns if c not in lead]]
    return cars

def suggest_car_randomized(customer: pd.Series, cars: pd.DataFrame,
                           sal_p10: float, sal_p90: float,
                           topk: int = 5, tie_eps: float = 0.03,
                           per_customer_random: bool = True) -> str | None:
    """
    STRICT selection (segment+style), randomize among near-ties.
    - Salary is clamped to [P10, P90] for realistic segmenting/budget.
    - topk: limit near-tie pool size.
    - tie_eps: cars within ±3% of best score count as ties.
    - per_customer_random: stable randomness seeded by CustomerId.
    """
    salary = float(np.clip(customer["EstimatedSalary"], sal_p10, sal_p90))
    age_bucket = str(customer.get("AgeBucket", ""))

    # salary → segment
    need_segment = "Economy" if salary < 50_000 else ("Mid" if salary < 150_000 else "Luxury")

    # age → preferred styles
    if age_bucket in ["<=25", "26-35"]:
        styles = ["Sport", "Compact"]
    elif age_bucket in ["46-55", "56-65", "65+"]:
        styles = ["Family", "General"]
    else:
        styles = ["General", "Compact", "Family", "Sport"]

    pool = cars[(cars["segment"] == need_segment) & (cars["style"].isin(styles))]
    if pool.empty:
        return None

    # score by closeness to budget
    budget = 0.6 * salary
    pool = pool.assign(score = - (pool["price_numeric"] - budget).abs())
    ranked = pool.sort_values("score", ascending=False)
    best = ranked.iloc[0]["score"]
    # gather near-ties
    near = ranked[ranked["score"] >= best - abs(best) * tie_eps]
    if near.empty:
        near = ranked.head(topk)
    else:
        near = near.head(max(topk, len(near)))

    # randomized pick (stable per customer unless you set per_customer_random=False)
    if per_customer_random:
        seed = int(customer.get("CustomerId", 0)) % (2**32 - 1)
        pick = near.sample(1, random_state=seed).iloc[0]["car_name"]
    else:
        pick = near.sample(1).iloc[0]["car_name"]
    return pick

# ---------- NEW: helpers to read & score any dataset ----------
def _read_any(file):
    name = getattr(file, "name", "")
    if str(name).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)

def score_raw_customers(raw_df: pd.DataFrame, th: float) -> pd.DataFrame:
    """
    Take a raw-like customers df (same columns as data.csv, 'Exited' optional),
    engineer features exactly like training, align columns to the model,
    predict churn prob at threshold th, and return a dataframe with predictions.
    """
    cust_ids = raw_df["CustomerId"].values if "CustomerId" in raw_df.columns else np.arange(len(raw_df))
    df_model = raw_df.drop(columns=["RowNumber", "Surname", "CustomerId"], errors="ignore")
    df_fe_local, _, num_cols_local = featurize(df_model)
    df_fe_local = iqr_cap(df_fe_local, num_cols_local)

    X_up = df_fe_local.drop(columns=["Exited"], errors="ignore")
    # uses global cat_cols/feature_cols defined later
    X_up_rf = pd.get_dummies(X_up, columns=cat_cols, drop_first=True)
    X_up_rf = X_up_rf.reindex(columns=feature_cols, fill_value=0)

    proba = rf.predict_proba(X_up_rf)[:, 1]
    pred  = (proba >= th).astype(int)

    df_fe_local["CustomerId"] = cust_ids
    df_fe_local["ChurnProb"]  = proba
    df_fe_local["ChurnPred"]  = pred
    return df_fe_local

# ------------------------------
# Caching layers
# ------------------------------
@st.cache_data
def load_data():
    # Keep raw for CustomerId & display; build modeling df from it
    df_raw = pd.read_csv("data.csv")
    # drop only non-predictive IDs from modeling copy
    df = df_raw.drop(columns=["RowNumber", "Surname", "CustomerId"])
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RSEED, stratify=y
    )
    X_train_rf, X_test_rf = ohe_align(X_train, X_test, cat_cols)
    rf = train_rf(X_train_rf, y_train)

    # Evaluate
    proba_test = rf.predict_proba(X_test_rf)[:, 1]
    pred_test = (proba_test >= 0.50).astype(int)
    metrics = summarize(y_test, pred_test, proba_test)

    # Save feature names for later alignment on full data
    feature_cols = X_train_rf.columns.tolist()
    return rf, feature_cols, metrics, (X_train, X_test, y_train, y_test)

# ------------------------------
# UI
# ------------------------------
st.title("Churn Prediction → Car Suggestions")

# Load data + model
df_raw, df_model = load_data()
df_fe, X_all, y_all, cat_cols = prepare_data(df_model)
rf, feature_cols, metrics, split_objs = fit_model(X_all, y_all, cat_cols)

# Salary clamps for realistic segmenting/budget
SAL_P10 = float(df_fe["EstimatedSalary"].quantile(0.10))
SAL_P90 = float(df_fe["EstimatedSalary"].quantile(0.90))

st.subheader("Model (Random Forest) — quick summary")
st.write(metrics)

with st.expander("Preview customer data (engineered)"):
    st.dataframe(df_fe.head(10))

# Threshold selection
st.subheader("Choose churn threshold")
th = st.slider("Decision threshold for P(churn)", min_value=0.05, max_value=0.95, value=0.30, step=0.01)

# Prepare full design matrix and predict on all rows of data.csv
X_full = df_fe.drop(columns=["Exited"])
X_full_rf = pd.get_dummies(X_full, columns=cat_cols, drop_first=True)
X_full_rf = X_full_rf.reindex(columns=feature_cols, fill_value=0)

proba_full = rf.predict_proba(X_full_rf)[:, 1]
pred_full  = (proba_full >= th).astype(int)

# === Threshold-aware summary (current slider) ===
st.subheader(f"Model summary (threshold = {th:.2f})")
if "Exited" in df_fe.columns:
    metrics_at_th = summarize(df_fe["Exited"], pred_full, proba_full)
    st.json(metrics_at_th)
else:
    st.info("Labels not available to compute metrics at the current threshold.")

# Attach predictions back to a copy for display (for the base data.csv)
pred_df = df_fe.copy()
pred_df["ChurnProb"] = proba_full
pred_df["ChurnPred"] = pred_full
pred_df["CustomerId"] = df_raw["CustomerId"]  # bring back id

# ==========================================
# PICK WHO TO TARGET  +  (optional) UPLOAD
# ==========================================
X_train, X_test, y_train, y_test = split_objs

st.subheader("Who should we target?")
scope = st.radio(
    "Population to score/display",
    ["All customers (data.csv)", "Hold-out test only", "Training only", "Upload a file"],
    index=0,
    horizontal=False
)
exclude_churned = st.checkbox("Exclude customers who already churned (Exited = 1)", value=True)

pred_df_source = None

if scope == "Upload a file":
    up = st.file_uploader("Upload current customers (.csv or .xlsx) with the same columns as data.csv", type=["csv","xlsx","xls"])
    if up is not None:
        try:
            raw_up = _read_any(up)
            pred_df_source = score_raw_customers(raw_up, th)
            st.success(f"Uploaded file read: {raw_up.shape[0]} rows.")
        except Exception as e:
            st.error(f"Could not read or score the uploaded file: {e}")
else:
    # use the already-scored base table
    if scope == "Hold-out test only":
        pred_df_source = pred_df.loc[X_test.index].copy()
    elif scope == "Training only":
        pred_df_source = pred_df.loc[X_train.index].copy()
    else:
        pred_df_source = pred_df.copy()

# optionally exclude historically churned customers if labels exist
if exclude_churned and "Exited" in pred_df_source.columns:
    pred_df_source = pred_df_source[pred_df_source["Exited"] == 0]

# filter at-risk
churners = pred_df_source[pred_df_source["ChurnPred"] == 1].copy()

# ------------------------------
# Cars + suggestions + display
# ------------------------------
st.subheader("Car suggestions for at-risk customers")
cars = load_and_label_cars("auto_combined_updated.xlsx")

with st.expander("Cars: derived labels preview"):
    st.write("Segment counts:", cars["segment"].value_counts(dropna=False))
    st.write("Style counts:", cars["style"].value_counts(dropna=False))
    st.dataframe(cars[["car_name","segment","style","price_numeric"]].head(20))

if churners.empty:
    st.info("No churn-risk customers for the selected population & threshold.")
else:
    churners["SuggestedCar"] = churners.apply(
        lambda r: suggest_car_randomized(r, cars, SAL_P10, SAL_P90,
                                         topk=5, tie_eps=0.03, per_customer_random=True),
        axis=1
    )
    churners = churners[churners["SuggestedCar"].notna()]

    show_cols = ["CustomerId", "Geography", "Gender", "Age", "AgeBucket",
                 "EstimatedSalary", "ChurnProb", "SuggestedCar"]
    show_cols = [c for c in show_cols if c in churners.columns]
    st.dataframe(churners[show_cols].sort_values("ChurnProb", ascending=False).head(50))

    # Download button
    out = churners[show_cols].sort_values("ChurnProb", ascending=False).reset_index(drop=True)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        out.to_excel(writer, index=False, sheet_name="churn_car_suggestions")
    st.download_button(
        label="Download suggestions (Excel)",
        data=buffer.getvalue(),
        file_name="churn_customers_with_car_suggestions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.caption("Tip: lower the threshold to increase recall (catch more churners), raise it to increase precision.")
