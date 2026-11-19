# MainCrafts AI & ML Internship - Task 3

## Model Validation, Overfitting Control & Hyperparameter Tuning

### Objective
Upgrade Task 2 California Housing prediction by adding cross-validation, systematic hyperparameter tuning, overfitting analysis and justified model selection.

### Dataset
- California Housing Dataset (same dataset as Task 2)
- Target: `MedHouseVal` (median house value)
- Input features: income, rooms, population, location and other dataset attributes

### Methods
- Train/test split
- 5-fold cross-validation
- GridSearchCV tuning
- Train/test gap analysis
- Model selection based on cross-validation RMSE

### Models
Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting.

### Evaluation metrics
MAE, RMSE, R²; cross-validation RMSE mean and standard deviation.

### Deliverables
Comparison tables, tuning results, overfitting report, graphs, saved best estimator, and final Markdown report.

### Conclusion
This workflow prioritizes reliable performance on unseen data and reproducible model selection over a single training score.
