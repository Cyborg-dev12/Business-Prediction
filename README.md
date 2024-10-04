# Advanced Business & Financial Prediction App

This project is a **Streamlit** app designed to predict future sales and financial expenses using advanced machine learning techniques like **Random Forest Regressor**. The app allows users to either manually input past sales and expenses data or upload a CSV file. It provides predictions for future months, along with evaluation metrics and visualizations.

## Features

- **Manual Data Entry**: Users can input past sales and expenses values manually and predict future values for a specified number of months.
- **CSV File Upload**: Users can upload CSV files containing past sales and expenses data, and the app will make predictions based on the uploaded data.
- **Machine Learning**: The app uses **Random Forest Regressor** to predict future sales and expenses.
- **Evaluation Metrics**: After training the model, the app displays important metrics like:
  - Mean Absolute Error (MAE)
  - R-squared (R²) score
- **Visualizations**: The app generates visual plots showing predicted sales and expenses over time.

## App Structure

- `app.py`: The main file containing the Streamlit app logic, model training, predictions, and visualizations.
- **Manual Input Mode**: Users can manually enter past data for sales and expenses and predict future trends.
- **CSV Upload Mode**: Users can upload a CSV file, and the app will read, process, and predict future data.
  
## How to Run the App Locally

### Clone the Repository
First, clone the GitHub repository to your local machine:

```bash
git clone https://github.com/Cyborg-dev12/Business-sales-prediction-app.git
