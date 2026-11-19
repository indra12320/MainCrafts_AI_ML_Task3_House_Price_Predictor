# MainCrafts Task 3 - Results

**Selected model:** Gradient Boosting  
**Selection method:** Lowest 5-fold CV RMSE on training data

## Hold-out test metrics
- MAE: 0.3484
- RMSE: 0.5114
- R²: 0.8004

## Selected hyperparameters
```json
{
  "model__learning_rate": 0.1,
  "model__max_depth": 3,
  "model__n_estimators": 200
}
```

## Model comparison
| Model             |   Selection_CV_RMSE |   Test_RMSE |   Test_MAE |   Test_R2 |
|:------------------|--------------------:|------------:|-----------:|----------:|
| Gradient Boosting |            0.504642 |    0.511381 |   0.348358 |  0.800436 |
| Random Forest     |            0.509701 |    0.504288 |   0.326793 |  0.805933 |
| Decision Tree     |            0.619904 |    0.602709 |   0.404523 |  0.72279  |
| Lasso             |            0.720499 |    0.744642 |   0.533145 |  0.576856 |
| Ridge             |            0.720509 |    0.745557 |   0.533193 |  0.575816 |
| Linear Regression |            0.72051  |    0.745581 |   0.5332   |  0.575788 |

## Overfitting analysis
| Model             |   Train_RMSE |   Test_RMSE |   RMSE_Gap_TestMinusTrain |   Train_R2 |   Test_R2 |   Test_MAE | Generalization_Notes                                           |
|:------------------|-------------:|------------:|--------------------------:|-----------:|----------:|-----------:|:---------------------------------------------------------------|
| Random Forest     |     0.185807 |    0.504288 |                 0.318482  |   0.974174 |  0.805933 |   0.326793 | Potential overfit: train-test gap is large                     |
| Gradient Boosting |     0.46843  |    0.511381 |                 0.0429515 |   0.835854 |  0.800436 |   0.348358 | Gap is relatively small; consider CV and test metrics together |
| Decision Tree     |     0.43959  |    0.602709 |                 0.16312   |   0.855444 |  0.72279  |   0.404523 | Gap is relatively small; consider CV and test metrics together |
| Lasso             |     0.719715 |    0.744642 |                 0.0249273 |   0.612509 |  0.576856 |   0.533145 | Gap is relatively small; consider CV and test metrics together |
| Ridge             |     0.719676 |    0.745557 |                 0.025881  |   0.612551 |  0.575816 |   0.533193 | Gap is relatively small; consider CV and test metrics together |
| Linear Regression |     0.719676 |    0.745581 |                 0.0259057 |   0.612551 |  0.575788 |   0.5332   | Gap is relatively small; consider CV and test metrics together |

## Limitations
- Dataset values are historical (1990 USD) and capped at the upper target range.
- Intended for learning, not current property valuation.
- CV/test scores may vary due to data split and sampling.
