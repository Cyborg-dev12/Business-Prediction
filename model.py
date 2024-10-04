import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = sns.load_dataset('tips')
X = data[['total_bill', 'size', 'sex', 'day']]
y = data['tip']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['total_bill', 'size']),
        ('cat', OneHotEncoder(drop='first'), ['sex', 'day'])  # drop first to avoid dummy variable trap
    ])
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

ridge = Ridge()
lasso = Lasso()

ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', poly),
    ('ridge', ridge)
])

lasso_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', poly),
    ('lasso', lasso)
])

param_grid = {
    'ridge__alpha': [0.1, 1.0, 10.0, 100.0],
    'lasso__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search_ridge = GridSearchCV(ridge_pipeline, {'ridge__alpha': param_grid['ridge__alpha']}, cv=kf, scoring='r2', n_jobs=-1)
grid_search_lasso = GridSearchCV(lasso_pipeline, {'lasso__alpha': param_grid['lasso__alpha']}, cv=kf, scoring='r2', n_jobs=-1)

grid_search_ridge.fit(X, y)
grid_search_lasso.fit(X, y)

best_ridge_model = grid_search_ridge.best_estimator_
best_lasso_model = grid_search_lasso.best_estimator_

best_ridge_alpha = grid_search_ridge.best_params_['ridge__alpha']
best_lasso_alpha = grid_search_lasso.best_params_['lasso__alpha']

print(f"Best alpha for Ridge: {best_ridge_alpha}")
print(f"Best alpha for Lasso: {best_lasso_alpha}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_ridge_model.fit(X_train, y_train)
best_lasso_model.fit(X_train, y_train)
y_pred_ridge = best_ridge_model.predict(X_test)
y_pred_lasso = best_lasso_model.predict(X_test)
def evaluate_model(y_test, y_pred, model_name):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = X_train.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print(f"\nModel: {model_name}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.2f}")
    print(f"Adjusted R-squared: {adjusted_r2:.2f}")


evaluate_model(y_test, y_pred_ridge, "Ridge")
evaluate_model(y_test, y_pred_lasso, "Lasso")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_ridge, color='blue', alpha=0.5, label='Ridge Predictions')
plt.scatter(y_test, y_pred_lasso, color='green', alpha=0.5, label='Lasso Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Fit Line')
plt.title('Predicted vs Actual Tips (Ridge vs Lasso)')
plt.xlabel('Actual Tips')
plt.ylabel('Predicted Tips')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_ridge, y_test - y_pred_ridge, color='blue', alpha=0.6, label='Ridge Residuals')
plt.scatter(y_pred_lasso, y_test - y_pred_lasso, color='green', alpha=0.6, label='Lasso Residuals')
plt.axhline(0, linestyle='--', color='red')
plt.title('Residuals (Ridge vs Lasso)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend()
plt.show()
