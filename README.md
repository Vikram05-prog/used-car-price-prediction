# Used Car Price Prediction

This project uses machine learning to estimate used-car prices from a CSV dataset.

## Files

- [Notebook](./Used_Car_Prediction_Model.ipynb) — EDA, cleaning, preprocessing, model comparison, feature engineering, and residual analysis.
- [Streamlit demo](./app.py) — upload a CSV, compare models, and try a price estimate.
- [requirements.txt](./requirements.txt) — Python libraries needed to run the demo.

## What the notebook does

1. Checks the dataset for missing values, duplicates, and unusual target values.
2. Plots the price distribution and explores relationships between features.
3. Fills missing numeric values with the median and missing categorical values with the most common value.
4. Scales numeric features and one-hot encodes categorical features using a pipeline.
5. Compares a median baseline, Linear Regression, Ridge, Lasso, and Random Forest.
6. If the dataset has a `make_year` column, compares it with a new `car_age` feature.
7. Plots residuals and displays the ten predictions with the largest errors.

The target column should be named `price_usd`. The notebook asks you to upload the CSV when you run it. The dataset is not included in this repository.

## Run the notebook

1. Open the [notebook in Google Colab](https://colab.research.google.com/github/Vikram05-prog/used-car-price-prediction/blob/main/Used_Car_Prediction_Model.ipynb).
2. Run the cells in order.
3. Upload your CSV when prompted.
4. Review the plots and the model comparison table.

## Run the demo locally

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/Vikram05-prog/used-car-price-prediction.git
cd used-car-price-prediction
python -m venv .venv
```

Activate the environment:

- Windows: `.venv\\Scripts\\activate`
- macOS/Linux: `source .venv/bin/activate`

Then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload the same type of CSV in the app. It displays the model scores and lets you enter feature values to get an estimate.

## Evaluation

The notebook reports MAE, RMSE, and R² on a held-out test set. Actual scores are not listed here because they depend on the dataset. Run the notebook first, then add the scores you get if you want to report them.

## Things to keep in mind

The model can make poor predictions for unusual cars or rows with incomplete or incorrect information. A random test split does not guarantee the same accuracy on newer listings. Check the dataset and the large prediction errors before drawing conclusions.

## Libraries

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and Streamlit.
