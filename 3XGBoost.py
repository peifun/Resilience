import pandas as pd
import xgboost as xgb
import numpy as np
import shap
import joblib
import matplotlib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

csv_path = r"E:\Unmanaged Forest.csv"
df = pd.read_csv(csv_path)
feature_cols = [
    'AT_CV', 'AT_MEAN', 'AT_TAC',
    'UMF_FVC',
    'PRE_CV', 'PRE_MEAN', 'PRE_TAC',
    'SSRD_CV', 'SSRD_MEAN', 'SSRD_TAC',
    'VPD_CV', 'VPD_MEAN', 'VPD_TAC'
]
target_col = 'Forest_TAC'
X = df[feature_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100
)
param_grid = {
    'max_depth': [20],
    'learning_rate': [0.1],
    'n_estimators': [2000],
    'reg_alpha': [0.1],
    'reg_lambda': [50]
}
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=100,
    n_jobs=-1
)
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

print(f"testing R²: {r2_score(y_test, y_pred_test):.4f}")

model_save_path = r"E:\xgboost_shap_Unmanaged Forest.csv"
joblib.dump(best_model, model_save_path)

scatter_df = pd.DataFrame({
    "Observed": y_test.values,
    "Predicted": y_pred_test
})
scatter_df.to_csv(
    r"E:\point_FVCUMF.csv",
    index=False
)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Mean_Abs_SHAP": mean_abs_shap
}).sort_values(by="Mean_Abs_SHAP", ascending=False)
shap_importance_df.to_csv(
    r"E:\Mean_Abs_SHAP_Contribution.csv",
    index=False
)
shap_cache_path = r"E:\xgb_shap_cacheUMF.pkl"
shap_cache = {
    "shap_values": shap_values,
    "feature_cols": feature_cols,
}
joblib.dump(shap_cache, shap_cache_path)

