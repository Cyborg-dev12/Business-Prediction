# not fully working completed yet
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def handle_data(uploaded_file):
    if uploaded_file is not None:
  
        df = pd.read_csv(uploaded_file)
    else:
        np.random.seed(42)
        date_range = pd.date_range(start='2020-01-01', periods=100, freq='M')
        sales = np.random.uniform(10000, 50000, size=(100,))
        expenses = np.random.uniform(5000, 30000, size=(100,))
        df = pd.DataFrame({'Date': date_range, 'Sales': sales, 'Expenses': expenses})
    
    return df

def preprocess_data(df):

    if 'Sales' not in df.columns:
        df['Sales'] = np.random.uniform(10000, 50000, size=len(df))
    
    if 'Expenses' not in df.columns:
        df['Expenses'] = np.random.uniform(5000, 30000, size=len(df))

    imputer = SimpleImputer(strategy='mean')
    df['Sales'] = imputer.fit_transform(df[['Sales']])
    df['Expenses'] = imputer.fit_transform(df[['Expenses']])
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day


        df = df.drop('Date', axis=1)
    else:
        st.warning("No 'Date' column found. Please ensure your data has a 'Date' column or adjust accordingly.")

    X = df.drop(['Sales', 'Expenses'], axis=1)  
    y_sales = df['Sales']
    y_expenses = df['Expenses']
    
    return X, y_sales, y_expenses

def build_and_train_model(X_train, y_train, X_val, y_val):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1) 
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, 
              batch_size=32, callbacks=[early_stopping], verbose=0)

    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return predictions
def plot_future_predictions(sales_predictions, expenses_predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(sales_predictions, label='Predicted Sales', color='blue')
    plt.plot(expenses_predictions, label='Predicted Expenses', color='orange')
    plt.title('Future Predictions')
    plt.xlabel('Months')
    plt.ylabel('Amount')
    plt.legend()
    plt.grid()
    st.pyplot(plt)

st.title("Business Sales and Expenses Prediction")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

st.subheader("Or enter data manually:")
num_records = st.number_input("Number of Records:", min_value=1, max_value=100, value=1)
sales_input = st.text_area("Sales (comma-separated):")
expenses_input = st.text_area("Expenses (comma-separated):")

df = handle_data(uploaded_file)

if uploaded_file is None and (sales_input or expenses_input):
    sales = list(map(float, sales_input.split(',')))
    expenses = list(map(float, expenses_input.split(',')))
    
    if len(sales) < num_records:
        sales += [np.random.uniform(10000, 50000) for _ in range(num_records - len(sales))]
    if len(expenses) < num_records:
        expenses += [np.random.uniform(5000, 30000) for _ in range(num_records - len(expenses))]
    
    date_range = pd.date_range(start='2020-01-01', periods=num_records, freq='M')
    df = pd.DataFrame({'Date': date_range, 'Sales': sales[:num_records], 'Expenses': expenses[:num_records]})

st.subheader("Data Preview:")
st.write(df)

X, y_sales, y_expenses = preprocess_data(df)

X_train, X_temp, y_sales_train, y_sales_temp, y_expenses_train, y_expenses_temp = train_test_split(
    X, y_sales, y_expenses, test_size=0.3, random_state=42)

X_val, X_test, y_sales_val, y_sales_test, y_expenses_val, y_expenses_test = train_test_split(
    X_temp, y_sales_temp, y_expenses_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

st.subheader("Training Models...")
sales_model = build_and_train_model(X_train, y_sales_train, X_val, y_sales_val)
expenses_model = build_and_train_model(X_train, y_expenses_train, X_val, y_expenses_val)
sales_predictions = evaluate_model(sales_model, X_test, y_sales_test)
expenses_predictions = evaluate_model(expenses_model, X_test, y_expenses_test)

st.subheader("Predicted Future Sales and Expenses")
plot_future_predictions(sales_predictions, expenses_predictions)
st.success("Prediction completed!")
