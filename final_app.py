#test1
# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# st.title("Advanced Business & Financial Prediction App")
# st.write("""This app allows you to predict sales and future financial expenses using advanced machine learning techniques. You can either enter data manually or upload a CSV file to make predictions.""")

# input_method = st.radio("Choose your input method:", ("Manual Input", "Upload CSV File"))

# def make_predictions(sales_input, expense_input, future_input):
#     # Generate random predictions based on the input sales and expenses
#     future_sales_pred = np.random.rand(future_input) * sales_input
#     future_expenses_pred = np.random.rand(future_input) * expense_input

#     total_sales_pred = np.sum(future_sales_pred)
#     total_expenses_pred = np.sum(future_expenses_pred)

#     profit_or_loss = total_sales_pred - total_expenses_pred
#     result = "Profit" if profit_or_loss > 0 else "Loss"

#     actual_sales = future_sales_pred * (1 + np.random.uniform(-0.1, 0.1, future_input))
#     actual_expenses = future_expenses_pred * (1 + np.random.uniform(-0.1, 0.1, future_input))

#     sales_accuracy = 100 * (1 - np.mean(np.abs((actual_sales - future_sales_pred) / actual_sales))) if np.any(actual_sales) else 0
#     expenses_accuracy = 100 * (1 - np.mean(np.abs((actual_expenses - future_expenses_pred) / actual_expenses))) if np.any(actual_expenses) else 0

#     return future_sales_pred, future_expenses_pred, total_sales_pred, total_expenses_pred, profit_or_loss, result, sales_accuracy, expenses_accuracy

# if input_method == "Manual Input":
#     st.subheader("Manually Enter Data:")
#     sales_input = st.number_input("Enter past sales data (in your currency):", format="%.2f", step=1000.00)
#     expense_input = st.number_input("Enter past expenses (in your currency):", format="%.2f", step=1000.00)
#     future_input = st.number_input("Enter the number of months to predict:", min_value=1, max_value=12)

#     if st.button("Predict Future Sales and Expenses"):
#         future_sales_pred, future_expenses_pred, total_sales_pred, total_expenses_pred, profit_or_loss, result, sales_accuracy, expenses_accuracy = make_predictions(sales_input, expense_input, future_input)

#         # Plot future predictions
#         months = np.arange(1, future_input + 1)
#         plt.figure(figsize=(10, 5))
#         plt.plot(months, future_sales_pred, label="Predicted Sales", color="blue", marker='o')
#         plt.plot(months, future_expenses_pred, label="Predicted Expenses", color="red", marker='o')
#         plt.title("Future Sales and Expenses Forecast")
#         plt.xlabel("Months")
#         plt.ylabel("Amount (in your currency)")
#         plt.xticks(months)
#         plt.legend()
#         st.pyplot(plt)

#         st.subheader("Prediction Results")
#         st.write(f"**Total Predicted Sales for {future_input} months:** {total_sales_pred:.2f}")
#         st.write(f"**Total Predicted Expenses for {future_input} months:** {total_expenses_pred:.2f}")
#         st.write(f"**Net Result (Total Sales - Total Expenses):** {profit_or_loss:.2f} ({result})")
#         st.write(f"**Sales Prediction Accuracy:** {sales_accuracy:.2f}%")
#         st.write(f"**Expenses Prediction Accuracy:** {expenses_accuracy:.2f}%")

# elif input_method == "Upload CSV File":
#     st.subheader("Upload Your CSV File:")
#     uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

#     if uploaded_file is not None:
#         # Read the CSV file
#         data = pd.read_csv(uploaded_file)
#         data.columns = [col.lower() for col in data.columns]
#         st.write("Uploaded Data:", data)  # Display the uploaded data for confirmation

#         # Default values if columns are missing
#         sales_input = data['sales'].sum() if 'sales' in data.columns else 0
#         expense_input = data['expenses'].sum() if 'expenses' in data.columns else 0
        
#         if 'sales' not in data.columns:
#             st.warning("Sales column is missing. Generating data for sales.")
#             data['sales'] = np.random.rand(data.shape[0]) * 1000  # Generate random sales data
#             sales_input = data['sales'].sum()  # Recalculate sales_input after generating data

#         if 'expenses' not in data.columns:
#             st.warning("Expenses column is missing. Generating data for expenses.")
#             data['expenses'] = np.random.rand(data.shape[0]) * 800  # Generate random expenses data
#             expense_input = data['expenses'].sum()  # Recalculate expense_input after generating data
        
#         future_input = data.shape[0]  # Assuming each row is a month of data

#         future_sales_pred, future_expenses_pred, total_sales_pred, total_expenses_pred, profit_or_loss, result, sales_accuracy, expenses_accuracy = make_predictions(sales_input, expense_input, future_input)

#         months = np.arange(1, future_input + 1)
#         plt.figure(figsize=(10, 5))
#         plt.plot(months, future_sales_pred, label="Predicted Sales", color="blue", marker='o')
#         plt.plot(months, future_expenses_pred, label="Predicted Expenses", color="red", marker='o')
#         plt.title("Future Sales and Expenses Forecast")
#         plt.xlabel("Months")
#         plt.ylabel("Amount (in your currency)")
#         plt.xticks(months)
#         plt.legend()
#         st.pyplot(plt)

#         st.subheader("Prediction Results")
#         st.write(f"**Total Predicted Sales for {future_input} months:** {total_sales_pred:.2f}")
#         st.write(f"**Total Predicted Expenses for {future_input} months:** {total_expenses_pred:.2f}")
#         st.write(f"**Net Result (Total Sales - Total Expenses):** {profit_or_loss:.2f} ({result})")
#         st.write(f"**Sales Prediction Accuracy:** {sales_accuracy:.2f}%")
#         st.write(f"**Expenses Prediction Accuracy:** {expenses_accuracy:.2f}%")

#final working test2
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def train_model(data):

    X = data[['month']] 
    y_sales = data['sales']
    y_expenses = data['expenses']

    X_train, X_test, y_train_sales, y_test_sales = train_test_split(X, y_sales, test_size=0.2, random_state=42)
    _, _, y_train_expenses, y_test_expenses = train_test_split(X, y_expenses, test_size=0.2, random_state=42)

    model_sales = RandomForestRegressor(n_estimators=100, random_state=42)
    model_sales.fit(X_train, y_train_sales)

    model_expenses = RandomForestRegressor(n_estimators=100, random_state=42)
    model_expenses.fit(X_train, y_train_expenses)
    sales_predictions = model_sales.predict(X_test)
    expenses_predictions = model_expenses.predict(X_test)

    sales_mae = mean_absolute_error(y_test_sales, sales_predictions)
    expenses_mae = mean_absolute_error(y_test_expenses, expenses_predictions)
    sales_r2 = r2_score(y_test_sales, sales_predictions)
    expenses_r2 = r2_score(y_test_expenses, expenses_predictions)

    return model_sales, model_expenses, sales_mae, expenses_mae, sales_r2, expenses_r2

def make_predictions(model_sales, model_expenses, months):
    future_months = np.array(months).reshape(-1, 1)
    future_sales_pred = model_sales.predict(future_months)
    future_expenses_pred = model_expenses.predict(future_months)
    return future_sales_pred, future_expenses_pred

st.title("Advanced Business & Financial Prediction App")
st.write("""This app allows you to predict sales and future financial expenses using advanced machine learning techniques. You can either enter data manually or upload a CSV file to make predictions.""")

input_method = st.radio("Choose your input method:", ("Manual Input", "Upload CSV File"))

if input_method == "Manual Input":
    st.subheader("Manually Enter Data:")
    sales_input = st.number_input("Enter past sales data (in your currency):", format="%.2f", step=1000.00)
    expense_input = st.number_input("Enter past expenses (in your currency):", format="%.2f", step=1000.00)
    future_input = st.number_input("Enter the number of months to predict:", min_value=1, max_value=12)

    if st.button("Train Model and Predict Future Sales and Expenses"):

        months = np.arange(1, 13).reshape(-1, 1)  
        sales_data = np.random.rand(12) * sales_input 
        expenses_data = np.random.rand(12) * expense_input  
        
        train_data = pd.DataFrame({
            'month': months.flatten(),
            'sales': sales_data,
            'expenses': expenses_data
        })

        model_sales, model_expenses, sales_mae, expenses_mae, sales_r2, expenses_r2 = train_model(train_data)

        future_months = np.arange(1, future_input + 1)
        future_sales_pred, future_expenses_pred = make_predictions(model_sales, model_expenses, future_months)

        st.subheader("Prediction Results")
        for month, sales, expenses in zip(future_months, future_sales_pred, future_expenses_pred):
            profit = sales - expenses
            st.write(f"**Predicted Sales for Month {month}:** {sales:.2f}")
            st.write(f"**Predicted Expenses for Month {month}:** {expenses:.2f}")
            st.write(f"**Predicted Profit for Month {month}:** {profit:.2f}")

        st.write(f"**Sales Model MAE:** {sales_mae:.2f}")
        st.write(f"**Expenses Model MAE:** {expenses_mae:.2f}")
        st.write(f"**Sales Model R²:** {sales_r2:.2f}")
        st.write(f"**Expenses Model R²:** {expenses_r2:.2f}")

        plt.figure(figsize=(10, 5))
        plt.plot(future_months, future_sales_pred, label="Predicted Sales", color="blue", marker='o')
        plt.plot(future_months, future_expenses_pred, label="Predicted Expenses", color="red", marker='o')
        plt.title("Future Sales and Expenses Forecast")
        plt.xlabel("Months")
        plt.ylabel("Amount (in your currency)")
        plt.xticks(future_months)
        plt.legend()
        st.pyplot(plt)

elif input_method == "Upload CSV File":
    st.subheader("Upload Your CSV File:")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)
        data.columns = [col.lower() for col in data.columns]
        st.write("Uploaded Data:", data)  
        sales_input = data['sales'].sum() if 'sales' in data.columns else 0
        expense_input = data['expenses'].sum() if 'expenses' in data.columns else 0
        
        if 'sales' not in data.columns:
            st.warning("Sales column is missing. Generating data for sales.")
            data['sales'] = np.random.rand(data.shape[0]) * 1000  
            sales_input = data['sales'].sum()  

        if 'expenses' not in data.columns:
            st.warning("Expenses column is missing. Generating data for expenses.")
            data['expenses'] = np.random.rand(data.shape[0]) * 800  
            expense_input = data['expenses'].sum()  

        data['month'] = np.arange(1, data.shape[0] + 1)
        model_sales, model_expenses, sales_mae, expenses_mae, sales_r2, expenses_r2 = train_model(data)

        future_months = np.arange(data.shape[0] + 1, data.shape[0] + 1 + 12).reshape(-1, 1)  # Next 12 months
        future_sales_pred, future_expenses_pred = make_predictions(model_sales, model_expenses, future_months)


        st.subheader("Prediction Results")
        for month, sales, expenses in zip(range(data.shape[0] + 1, data.shape[0] + 13), future_sales_pred, future_expenses_pred):
            profit = sales - expenses
            st.write(f"**Predicted Sales for Month {month}:** {sales:.2f}")
            st.write(f"**Predicted Expenses for Month {month}:** {expenses:.2f}")
            st.write(f"**Predicted Profit for Month {month}:** {profit:.2f}")


        st.write(f"**Sales Model MAE:** {sales_mae:.2f}")
        st.write(f"**Expenses Model MAE:** {expenses_mae:.2f}")
        st.write(f"**Sales Model R²:** {sales_r2:.2f}")
        st.write(f"**Expenses Model R²:** {expenses_r2:.2f}")

        plt.figure(figsize=(10, 5))
        plt.plot(range(data.shape[0] + 1, data.shape[0] + 13), future_sales_pred, label="Predicted Sales", color="blue", marker='o')
        plt.plot(range(data.shape[0] + 1, data.shape[0] + 13), future_expenses_pred, label="Predicted Expenses", color="red", marker='o')
        plt.title("Future Sales and Expenses Forecast")
        plt.xlabel("Months")
        plt.ylabel("Amount (in your currency)")
        plt.xticks(range(data.shape[0] + 1, data.shape[0] + 13))
        plt.legend()
        st.pyplot(plt)
