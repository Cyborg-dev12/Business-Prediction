import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('personal_finance_expense\\personal_expenses.csv')

X = data[['income', 'age', 'spending_category', 'location']]
y = data['expenses']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['income', 'age']),
        ('cat', OneHotEncoder(drop='first'), ['spending_category', 'location'])
    ])

elastic_net = ElasticNet()
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('elasticnet', elastic_net)])


param_grid = {
    'elasticnet__alpha': np.logspace(-4, 2, 10), 
    'elasticnet__l1_ratio': np.linspace(0.1, 0.9, 9) 
}

kf = KFold(n_splits=10, shuffle=True, random_state=42)

elastic_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='r2', n_jobs=-1, verbose=1)
elastic_search.fit(X, y)
best_elastic_net = elastic_search.best_estimator_

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_elastic_net.fit(X_train, y_train)
y_pred = best_elastic_net.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}, R²: {r2:.2f}")
print(f"Best ElasticNet Parameters: {elastic_search.best_params_}")


future_income = np.linspace(5000, 8000, 12)
future_age = np.repeat(30, 12)  
future_spending_category = np.repeat('groceries', 12) 
future_location = np.repeat('urban', 12) 


future_data = pd.DataFrame({
    'income': future_income,
    'age': future_age,
    'spending_category': future_spending_category,
    'location': future_location
})

future_expenses_pred = best_elastic_net.predict(future_data)

print("Predicted Future Expenses for the Next 12 Months:")
for month, expense in enumerate(future_expenses_pred, 1):
    print(f"Month {month}: Predicted Expense = ${expense:.2f}")

monthly_budget = 6000

print("\nBudgeting Outcome for the Next 12 Months:")
budgeting_outcome = []
for month, expense in enumerate(future_expenses_pred, 1):
    if expense > monthly_budget:
        budgeting_outcome.append(expense - monthly_budget)
        print(f"Month {month}: Over Budget by ${expense - monthly_budget:.2f}")
    else:
        budgeting_outcome.append(monthly_budget - expense)
        print(f"Month {month}: Under Budget by ${monthly_budget - expense:.2f}")


months = np.arange(1, 13)
plt.figure(figsize=(10, 6))
plt.plot(months, future_expenses_pred, marker='o', label='Predicted Expenses', color='blue')
plt.axhline(y=monthly_budget, color='red', linestyle='--', label='Monthly Budget ($6000)')
plt.title('Predicted Future Expenses vs Budget')
plt.xlabel('Month')
plt.ylabel('Expense ($)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(months, budgeting_outcome, color=['green' if x > 0 else 'red' for x in budgeting_outcome], label='Budget Outcome')
plt.axhline(0, color='black', linewidth=1)
plt.title('Monthly Budget Outcome (Under/Over Budget)')
plt.xlabel('Month')
plt.ylabel('Amount ($)')
plt.grid(True)
plt.show()
