"""Local Streamlit demo for used-car price prediction."""
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

TARGET, RANDOM_STATE = "price_usd", 42
st.set_page_config(page_title="Used Car Price Demo", page_icon="🚗", layout="wide")
st.title("🚗 Used Car Price Prediction")
st.write("Upload a CSV with a numeric `price_usd` column to compare models and try a price estimate.")
uploaded = st.file_uploader("Upload used-car dataset (CSV)", type=["csv"])
if uploaded is None:
    st.info("Upload your CSV to train and compare models.")
    st.stop()
try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()
if TARGET not in df.columns:
    st.error(f"Target column {TARGET!r} not found. Rename it or change TARGET in app.py.")
    st.stop()
df = df.drop_duplicates().copy()
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET])
df = df[np.isfinite(df[TARGET])]
if len(df) < 10:
    st.error("Please upload a dataset with at least 10 usable rows.")
    st.stop()
if "make_year" in df.columns:
    df["make_year"] = pd.to_numeric(df["make_year"], errors="coerce")
    df["car_age"] = date.today().year - df["make_year"]
    df = df.drop(columns=["make_year"])
X, y = df.drop(columns=[TARGET]), df[TARGET]
if X.shape[1] == 0:
    st.error("No feature columns remain after removing target.")
    st.stop()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=RANDOM_STATE)
nums = X_train.select_dtypes(include=np.number).columns.tolist()
cats = X_train.select_dtypes(exclude=np.number).columns.tolist()
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
pre = ColumnTransformer([("numeric", num_pipe, nums), ("categorical", cat_pipe, cats)])
estimators = {
    "Median baseline": DummyRegressor(strategy="median"),
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=.01, max_iter=10000),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=2)
}
pipelines, rows = {}, []
for name, estimator in estimators.items():
    model = Pipeline([("preprocessing", pre), ("model", estimator)])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pipelines[name] = model
    rows.append({"Model": name, "MAE": mean_absolute_error(y_test, pred),
                 "RMSE": np.sqrt(mean_squared_error(y_test, pred)), "R2": r2_score(y_test, pred)})
comparison = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
st.subheader("Held-out test-set comparison")
st.dataframe(comparison.round(3), use_container_width=True)
best_name = comparison.iloc[0]["Model"]
st.caption(f"Lowest RMSE on this test split: {best_name}.")
st.subheader("Estimate a vehicle price")
st.write("Enter feature values compatible with the uploaded dataset.")
with st.form("prediction_form"):
    values = {}
    left, right = st.columns(2)
    for i, col in enumerate(X.columns):
        with (left if i % 2 == 0 else right):
            if pd.api.types.is_numeric_dtype(X[col]):
                valid = pd.to_numeric(X[col], errors="coerce").dropna()
                values[col] = st.number_input(str(col), value=float(valid.median()) if len(valid) else 0.0)
            else:
                options = X[col].dropna().astype(str).value_counts().index.tolist() or ["Unknown"]
                values[col] = st.selectbox(str(col), options)
    submitted = st.form_submit_button("Estimate price")
if submitted:
    sample = pd.DataFrame([values], columns=X.columns)
    try:
        estimate = float(pipelines[best_name].predict(sample)[0])
        st.success(f"Estimated price: {estimate:,.2f} (dataset target units)")
        st.caption("This is an estimate based on the uploaded data.")
    except Exception as exc:
        st.error(f"Prediction failed; check dataset feature values. Details: {exc}")
with st.expander("Dataset audit"):
    st.write(f"Rows used: {len(df):,} | Features: {X.shape[1]}")
    st.dataframe(df.isna().sum().sort_values(ascending=False).to_frame("Missing values"))
    st.dataframe(y.describe().to_frame("price_usd"))
