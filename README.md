# MainCrafts AI & ML Internship - Task 3

## Model Validation, Overfitting Control & Hyperparameter Tuning

This project upgrades the Task 2 California Housing Price Prediction using the **same California Housing dataset**.

### Implemented
- 5-fold cross-validation comparing 6 regression models
- GridSearchCV tuning for Ridge, Lasso, Decision Tree, Random Forest and Gradient Boosting
- Overfitting/generalization analysis using train vs test RMSE and R²
- Final selection based on CV RMSE (test data held back for final evaluation)
- MAE, RMSE and R² evaluation
- CSV/JSON/Markdown reports and diagnostic graphs
- Saved best model pipeline using joblib

### Models
Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting.

### Windows PowerShell setup
Run each command on a separate line:
```powershell
py -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python house_price_task3.py
```

If activation fails, run directly with the venv's Python:
```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe house_price_task3.py
```

Internet is needed on first run to fetch the dataset through scikit-learn.

### Output files
`reports/` includes cross-validation results, tuning results, overfitting analysis, final model comparison, prediction residuals, JSON summary, report and graphs.
`artifacts/` contains the selected model (`best_model.joblib`) and feature names.

### Important
`MedHouseVal` is measured in $100,000 units (1990 USD), not modern listing prices. This is an educational model, not real-estate valuation advice.
