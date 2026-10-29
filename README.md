# Used Car Price Prediction

A beginner-friendly ML project that predicts used-car prices from a CSV. It includes data audit, EDA, baseline regression, reusable preprocessing, model comparison, a feature-engineering experiment, residual analysis, and a small demo app.

## Required deliverables
- **EDA notebook:** [Used_Car_Prediction_Model.ipynb](./Used_Car_Prediction_Model.ipynb)
- **Preprocessing pipeline:** median imputation + scaling for numeric columns; most-frequent imputation + one-hot encoding for categorical columns; combined using `ColumnTransformer` and `Pipeline`.
- **Regression comparison table:** median baseline, Linear Regression, Ridge, Lasso, and Random Forest; MAE, RMSE, and R² on held-out test data.
- **Feature-engineering experiment:** compares `make_year` with engineered `car_age` if the column exists.
- **Residual analysis:** residual plot, residual distribution, and ten largest absolute errors.
- **Demo:** [app.py](./app.py), a Streamlit app that accepts a CSV, compares models, and predicts from entered feature values.
- **README:** setup instructions, dataset expectations, workflow, and limitations.

## Dataset
The dataset is **not committed** to this repository. Use a CSV you are permitted to use. By default, the numeric target column must be named `price_usd`. Other feature columns are inferred automatically. If `make_year` exists, the notebook compares it with `car_age`; otherwise that experiment is skipped.

## Run the notebook in Google Colab
1. Open [the notebook in Colab](https://colab.research.google.com/github/Vikram05-prog/used-car-price-prediction/blob/main/Used_Car_Prediction_Model.ipynb).
2. Run cells from top to bottom.
3. Upload the dataset CSV when prompted.
4. Review EDA plots, model metrics, feature-engineering results, and residual analysis.

## Run the demo locally
Python 3.10+ recommended.

```bash
git clone https://github.com/Vikram05-prog/used-car-price-prediction.git
cd used-car-price-prediction
python -m venv .venv
```

Activate the environment (Windows: `.venv\\Scripts\\activate`; macOS/Linux: `source .venv/bin/activate`), then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload your CSV in the browser. The app trains models on that file and presents metrics and a prediction form. It retrains on each upload and is intended for local demonstration only.

## Metrics
Metric values are generated from the uploaded CSV and are intentionally not hard-coded here. After running the notebook, you can add your actual dataset details and observed results.
- **MAE:** average absolute error, in the target's units.
- **RMSE:** gives larger errors more weight.
- **R²:** compares model fit with a mean-target reference and can be negative.

## Limitations
- Results depend on data quality, target definition, geography, currency, and listing date.
- A random split estimates performance on this dataset, not necessarily future prices.
- Extreme prices may be valid rare vehicles or data errors; investigate before removing.
- This is an educational demo, not a professional appraisal or production service.

## Tech stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit.
