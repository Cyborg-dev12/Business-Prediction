import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000
income = np.random.uniform(3000, 15000, n_samples)  
age = np.random.randint(20, 70, n_samples)  
spending_category = np.random.choice(['groceries', 'utilities', 'entertainment', 'healthcare'], n_samples)
location = np.random.choice(['urban', 'suburban', 'rural'], n_samples) 
expenses = income * np.random.uniform(0.2, 0.8, n_samples) + np.random.normal(500, 200, n_samples)  
data = pd.DataFrame({
    'income': income,
    'age': age,
    'spending_category': spending_category,
    'location': location,
    'expenses': expenses  
})
data.to_csv('personal_expenses.csv', index=False)
print("Synthetic personal finance dataset created and saved as 'personal_expenses.csv'.")
