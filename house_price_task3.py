from pathlib import Path
import json, warnings
import joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
ROOT=Path(__file__).resolve().parent
REPORTS=ROOT/"reports"; GRAPHS=REPORTS/"graphs"; ARTIFACTS=ROOT/"artifacts"
for p in (REPORTS,GRAPHS,ARTIFACTS): p.mkdir(parents=True,exist_ok=True)
SEED=42; FOLDS=5

def pipeline(estimator, scale=False):
    steps=[("imputer",SimpleImputer(strategy="median"))]
    if scale: steps.append(("scaler",StandardScaler()))
    steps.append(("model",estimator))
    return Pipeline(steps)

def rmse(y,p): return float(np.sqrt(mean_squared_error(y,p)))
def metrics(y,p): return {"MAE":float(mean_absolute_error(y,p)),"RMSE":rmse(y,p),"R2":float(r2_score(y,p))}

def main():
    print("Loading California Housing dataset...")
    dataset=fetch_california_housing(as_frame=True)
    df=dataset.frame.copy()
    X=df.drop(columns=["MedHouseVal"]); y=df["MedHouseVal"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=SEED)
    cv=KFold(n_splits=FOLDS,shuffle=True,random_state=SEED)

    specs={
      "Linear Regression":(pipeline(LinearRegression(),True),None),
      "Ridge":(pipeline(Ridge(),True),{"model__alpha":[0.1,1,10,100]}),
      "Lasso":(pipeline(Lasso(max_iter=10000),True),{"model__alpha":[0.001,0.01,0.1,1]}),
      "Decision Tree":(pipeline(DecisionTreeRegressor(random_state=SEED)),{"model__max_depth":[None,5,10,20],"model__min_samples_leaf":[1,5,10]}),
      "Random Forest":(pipeline(RandomForestRegressor(random_state=SEED,n_jobs=-1)),{"model__n_estimators":[100,200],"model__max_depth":[None,15],"model__min_samples_leaf":[1,3]}),
      "Gradient Boosting":(pipeline(GradientBoostingRegressor(random_state=SEED)),{"model__n_estimators":[100,200],"model__learning_rate":[0.05,0.1],"model__max_depth":[2,3]})
    }

    print("\n[1/4] Five-fold baseline cross-validation...")
    cv_rows=[]
    cv_lookup={}
    for name,(est,grid) in specs.items():
        result=cross_validate(est,X_train,y_train,cv=cv,n_jobs=-1,
          scoring={"rmse":"neg_root_mean_squared_error","mae":"neg_mean_absolute_error","r2":"r2"},
          return_train_score=True)
        row={"Model":name,"CV_RMSE_mean":float(-result["test_rmse"].mean()),
             "CV_RMSE_std":float(result["test_rmse"].std()),
             "CV_MAE_mean":float(-result["test_mae"].mean()),"CV_R2_mean":float(result["test_r2"].mean()),
             "Train_RMSE_mean":float(-result["train_rmse"].mean()),"Train_R2_mean":float(result["train_r2"].mean())}
        cv_rows.append(row); cv_lookup[name]=row["CV_RMSE_mean"]
        print(f"{name}: CV RMSE={row['CV_RMSE_mean']:.4f} (+/- {row['CV_RMSE_std']:.4f})")
    cv_df=pd.DataFrame(cv_rows).sort_values("CV_RMSE_mean")
    cv_df.to_csv(REPORTS/"cross_validation_results.csv",index=False)

    print("\n[2/4] GridSearchCV hyperparameter tuning...")
    tuned_models={}; tuning_rows=[]; details={}
    for name,(est,grid) in specs.items():
        if grid is None:
            est.fit(X_train,y_train); tuned_models[name]=est
            pred=est.predict(X_train)
            tuning_rows.append({"Model":name,"Tuned":"No (baseline)","Best_Params":"{}","Best_CV_RMSE":np.nan,
              "Train_RMSE":rmse(y_train,pred),"Train_R2":float(r2_score(y_train,pred))})
            details[name]={"params":{},"cv_rmse":cv_lookup[name]}
            continue
        search=GridSearchCV(est,grid,cv=cv,scoring="neg_root_mean_squared_error",n_jobs=-1,refit=True,return_train_score=True)
        search.fit(X_train,y_train)
        tuned_models[name]=search.best_estimator_
        train_pred=search.best_estimator_.predict(X_train)
        tuning_rows.append({"Model":name,"Tuned":"Yes","Best_Params":json.dumps(search.best_params_),
          "Best_CV_RMSE":float(-search.best_score_),"Train_RMSE":rmse(y_train,train_pred),"Train_R2":float(r2_score(y_train,train_pred))})
        details[name]={"params":search.best_params_,"cv_rmse":float(-search.best_score_)}
        pd.DataFrame(search.cv_results_).to_csv(REPORTS/f"gridsearch_details_{name.lower().replace(' ','_')}.csv",index=False)
        print(f"{name}: best CV RMSE={-search.best_score_:.4f}; {search.best_params_}")
    pd.DataFrame(tuning_rows).to_csv(REPORTS/"hyperparameter_tuning_results.csv",index=False)

    print("\n[3/4] Train/test overfitting analysis and final model evaluation...")
    overfit_rows=[]; selection_rows=[]
    for name,est in tuned_models.items():
        train_pred=est.predict(X_train); test_pred=est.predict(X_test)
        train_rmse=rmse(y_train,train_pred); test_m=metrics(y_test,test_pred)
        gap=test_m["RMSE"]-train_rmse
        overfit_rows.append({"Model":name,"Train_RMSE":train_rmse,"Test_RMSE":test_m["RMSE"],
          "RMSE_Gap_TestMinusTrain":gap,"Train_R2":float(r2_score(y_train,train_pred)),
          "Test_R2":test_m["R2"],"Test_MAE":test_m["MAE"],
          "Generalization_Notes":"Potential overfit: train-test gap is large" if gap>0.25 else "Gap is relatively small; consider CV and test metrics together"})
        selection_rows.append({"Model":name,"Selection_CV_RMSE":details[name]["cv_rmse"],
          "Test_RMSE":test_m["RMSE"],"Test_MAE":test_m["MAE"],"Test_R2":test_m["R2"]})
    overfit=pd.DataFrame(overfit_rows).sort_values("Test_RMSE")
    overfit.to_csv(REPORTS/"overfitting_analysis.csv",index=False)
    comparison=pd.DataFrame(selection_rows).sort_values("Selection_CV_RMSE")
    comparison.to_csv(REPORTS/"final_model_comparison.csv",index=False)

    winner_name=comparison.iloc[0]["Model"]
    winner=tuned_models[winner_name]
    pred=winner.predict(X_test); final_metrics=metrics(y_test,pred)
    joblib.dump(winner,ARTIFACTS/"best_model.joblib")
    joblib.dump(list(X.columns),ARTIFACTS/"feature_names.joblib")
    (REPORTS/"best_model_summary.json").write_text(json.dumps({"selected_model":winner_name,
      "selection_method":"Lowest CV RMSE using training data","best_parameters":details[winner_name]["params"],
      "test_metrics":final_metrics,"target_scale":"MedHouseVal is in units of $100,000 (1990 USD)"},indent=2))

    print("\n[4/4] Generating plots and reports...")
    def bar_plot(labels,values,title,ylabel,path,errors=None):
        plt.figure(figsize=(10,6)); plt.bar(labels,values,yerr=errors,capsize=4)
        plt.xticks(rotation=25,ha="right"); plt.ylabel(ylabel); plt.title(title); plt.tight_layout(); plt.savefig(path,dpi=160); plt.close()
    sorted_cv=cv_df.sort_values("CV_RMSE_mean")
    bar_plot(sorted_cv.Model,sorted_cv.CV_RMSE_mean,"Baseline Models: 5-Fold CV RMSE","CV RMSE",GRAPHS/"baseline_cv_rmse.png",sorted_cv.CV_RMSE_std)
    plt.figure(figsize=(10,6)); idx=np.arange(len(overfit)); width=0.38
    plt.bar(idx-width/2,overfit.Train_RMSE,width,label="Train RMSE"); plt.bar(idx+width/2,overfit.Test_RMSE,width,label="Test RMSE")
    plt.xticks(idx,overfit.Model,rotation=25,ha="right"); plt.ylabel("RMSE"); plt.title("Overfitting Analysis: Train vs Test"); plt.legend(); plt.tight_layout()
    plt.savefig(GRAPHS/"train_vs_test_rmse.png",dpi=160); plt.close()
    sorted_final=comparison.sort_values("Selection_CV_RMSE")
    bar_plot(sorted_final.Model,sorted_final.Selection_CV_RMSE,"Tuned Model Selection by CV RMSE","CV RMSE",GRAPHS/"tuned_model_cv_rmse.png")
    plt.figure(figsize=(6,6)); plt.scatter(y_test,pred,alpha=0.25); lim=[min(float(y_test.min()),float(pred.min())),max(float(y_test.max()),float(pred.max()))]
    plt.plot(lim,lim,linestyle="--"); plt.xlabel("Actual median value ($100,000s)"); plt.ylabel("Predicted median value ($100,000s)")
    plt.title(f"Actual vs Predicted - {winner_name}"); plt.tight_layout(); plt.savefig(GRAPHS/"best_model_actual_vs_predicted.png",dpi=160); plt.close()
    residual=y_test-pred
    plt.figure(figsize=(8,5)); plt.scatter(pred,residual,alpha=0.25); plt.axhline(0,linestyle="--")
    plt.xlabel("Predicted ($100,000s)"); plt.ylabel("Residual"); plt.title(f"Residual Plot - {winner_name}")
    plt.tight_layout(); plt.savefig(GRAPHS/"best_model_residuals.png",dpi=160); plt.close()
    pd.DataFrame({"Actual":y_test,"Predicted":pred,"Residual":residual}).to_csv(REPORTS/"best_model_test_predictions.csv",index=False)

    report=f"""# MainCrafts Task 3 - Results

**Selected model:** {winner_name}  
**Selection method:** Lowest 5-fold CV RMSE on training data

## Hold-out test metrics
- MAE: {final_metrics['MAE']:.4f}
- RMSE: {final_metrics['RMSE']:.4f}
- R²: {final_metrics['R2']:.4f}

## Selected hyperparameters
```json
{json.dumps(details[winner_name]['params'],indent=2)}
```

## Model comparison
{comparison.to_markdown(index=False)}

## Overfitting analysis
{overfit.to_markdown(index=False)}

## Limitations
- Dataset values are historical (1990 USD) and capped at the upper target range.
- Intended for learning, not current property valuation.
- CV/test scores may vary due to data split and sampling.
"""
    (REPORTS/"task3_report.md").write_text(report,encoding="utf-8")
    print("\nDone!")
    print(f"Selected model: {winner_name}")
    print(f"Test MAE={final_metrics['MAE']:.4f} | RMSE={final_metrics['RMSE']:.4f} | R2={final_metrics['R2']:.4f}")
    print(f"Outputs: {REPORTS}")
    print(f"Saved model: {ARTIFACTS/'best_model.joblib'}")

if __name__=="__main__": main()
