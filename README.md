# Used Car Prediction Model

A beginner-friendly machine learning project to predict used-car prices from a Kaggle CSV dataset.

## Notebook
Open **[Used_Car_Prediction_Model.ipynb](./Used_Car_Prediction_Model.ipynb)** in Google Colab or Jupyter Notebook.

## Project workflow
1. Load and inspect the CSV dataset.
2. Perform exploratory data analysis (EDA).
3. Engineer `car_age` from `make_year`.
4. Split features and target, then create a train/test split.
5. Handle missing numerical values with median imputation.
6. Handle missing categorical values with most-frequent imputation.
7. Encode categorical features using one-hot encoding.
8. Use `ColumnTransformer` and `Pipeline` to combine preprocessing and modeling.
9. Train and compare Linear Regression and Random Forest Regression.
10. Evaluate with MAE, RMSE, and R².

## Dataset
The notebook expects a CSV with a numeric target column named `price_usd`. Upload your Kaggle CSV when prompted in Colab. The dataset itself is not included in this repository.

## Run in Google Colab
1. Open the notebook from this repository.
2. Run cells from top to bottom.
3. Upload the dataset CSV when asked.
4. Review the EDA graphs and model metrics.

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

## Learning level
Beginner / early machine learning. The notebook focuses on core preprocessing and regression concepts rather than advanced modeling.

## Limitations
This is an educational project, not a production pricing service. Results depend on dataset quality, market, currency, and train/test split. Good test metrics do not guarantee real-world accuracy.
