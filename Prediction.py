import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
import plotly.graph_objects as go

sugar_cane_data = pd.DataFrame({
    'Raw sugar import': [1689563.00, 1721797.00, 1943752.40, 1776451.45, 1860986.62, 1940346.50, 1944764.74, 1727585.00, 2079977.73, 1887093.04],
    'Raw sugar processed': [1737406.00, 1749858.00, 1971901.40, 1778548.95, 1862835.69, 1942630.81, 1946223.99, 1728762.96, 2078810.47, 1889095.14],
    'Raw sugar produced': [47843.00, 28072.00, 28149.00, 2097.58, 1849.07, 2284.31, 1502.28, 1177.96, 1802.74, 2002.10],
    'Import Quantity': [0.00, 45.00, 58.24, 115.86, 727.25, 735.81, 65.74, 124.77, 185.48, 154.96],
    'Loss': [7861.58, 5318.64, 491.98, 1628.15, 1508.05, 1613.04, 1583.08, 1166.26, 1316.05, 1496.21],
    'Processed': [340035.54, 199943.02, 200491.45, 14940.00, 13170.00, 16270.00, 10700.00, 8390.00, 12840.00, 14260.00],
    'Production': [146164.00, 98885.18, 9147.00, 30271.00, 28038.00, 29990.00, 23475.04, 20761.12, 23519.41, 24931.35],
    'Area harvested': [4346.0, 2847.0, 237.0, 1717.0, 1515.0, 1668.0, 1400.0, 1174.0, 1311.0, 1326.0],
    'Yield': [336318.0, 347336.0, 385949.0, 176316.0, 185009.0, 179770.0, 167675.0, 176784.0, 179453.0, 188035.0],
    'Burning crop residues (Emissions CH4)': [212.8, 140.0, 11.2, 84.0, 75.6, 81.2, 70.0, 58.8, 64.4, 64.4],
    'Burning crop residues (Emissions N2O)': [53.0, 26.5, 0.0, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5],
    'Total CO2eq emissions': [265.8, 166.5, 11.2, 110.5, 102.1, 107.7, 96.5, 85.3, 90.9, 90.9],
    'Total supply': [146164.00, 98930.18, 9205.24, 30386.86, 28765.25, 30725.81, 23540.78, 20885.89, 23704.89, 25086.31]
}, index=[2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021])

# Define target and features
target_variable = 'Raw sugar produced'
features = sugar_cane_data.columns.difference([target_variable])

# Splitting the data
X = sugar_cane_data[features]
y = sugar_cane_data[target_variable]

# Streamlit user input for future years
st.title("Sugar Supply Chain Sustainability Forecast")
st.write("Please input the future data for prediction:")

# User input for future years data
future_years_input = st.text_input("Enter future years separated by commas", "2022, 2023, 2024, 2025, 2026")
future_years = list(map(int, future_years_input.split(',')))

# Default values for the prediction data
default_values = {
    'Raw sugar import': [1840346.50, 1744764.74, 1627585.00, 1579977.73, 1487093.04],
    'Raw sugar produced': [2284.31, 2502.28, 2777.96, 2802.74, 3002.10],
    'Raw sugar processed': [1942630.81, 1946223.99, 1728762.96, 2078810.47, 1889095.14],
    'Import Quantity': [35.81, 65.74, 24.77, 85.48, 54.96],
    'Loss': [1613.04, 1583.08, 1166.26, 1316.05, 1496.21],
    'Processed': [16270.00, 10700.00, 8390.00, 12840.00, 14260.00],
    'Production': [29990.00, 33475.04, 36761.12, 38519.41, 40931.35],
    'Area harvested': [1668.0, 1400.0, 1174.0, 1311.0, 1326.0],
    'Yield': [179770.0, 167675.0, 176784.0, 179453.0, 188035.0],
    'Burning crop residues (Emissions CH4)': [91.2, 100.0, 118.8, 124.4, 134.4],
    'Burning crop residues (Emissions N2O)': [36.5, 46.5, 56.5, 66.5, 76.5],
    'Total CO2eq emissions': [117.7, 126.5, 135.3, 140.9, 150.9],
    'Total supply': [31725.81, 32540.78, 33885.89, 34704.89, 35086.31]
}

# Create empty DataFrame with future years as index
to_predict_data = pd.DataFrame(index=future_years)

# User inputs for each feature
for feature in features:
    # Join the default values as a string to show in the input box
    default_value_str = ", ".join(map(str, default_values.get(feature, [])))
    # Get user input and use the default values as initial values
    feature_input = st.text_input(f"Enter values for {feature} separated by commas", default_value_str)
    if feature_input:
        values = list(map(float, feature_input.split(',')))
        to_predict_data[feature] = values
    else:
        st.error(f"Please enter values for {feature}.")

# Check if data is ready for prediction
if not to_predict_data.isnull().values.any():
    pass
else:
    st.error("Please provide valid input data for all features.")

# Button to perform prediction
if st.button('Predict with Matplotlib'):
    # Train models with best parameters and make predictions
    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    future_predictions_linear = linear_model.predict(to_predict_data[features])

    # Lasso Regression
    lasso_model = Lasso(alpha=0.001)
    lasso_model.fit(X, y)
    future_predictions_lasso = lasso_model.predict(to_predict_data[features])

    # Random Forest
    rf_model = RandomForestRegressor(max_depth=None, min_samples_split=2, n_estimators=300)
    rf_model.fit(X, y)
    future_predictions_rf = rf_model.predict(to_predict_data[features])

    # Gradient Boosting
    gb_model = GradientBoostingRegressor(learning_rate=0.2, max_depth=3, n_estimators=200)
    gb_model.fit(X, y)
    future_predictions_gb = gb_model.predict(to_predict_data[features])

    # Plotting the original data and future predictions
    fig, ax = plt.subplots()
    ax.plot(sugar_cane_data.index, y, label='Original Data', linestyle='-', marker='o')
    ax.plot(to_predict_data.index, future_predictions_linear, label='Linear Prediction', linestyle='--', marker='x')
    ax.plot(to_predict_data.index, future_predictions_lasso, label='Lasso Prediction', linestyle='--', marker='x')
    ax.plot(to_predict_data.index, future_predictions_rf, label='Random Forest Prediction', linestyle='--', marker='x')
    ax.plot(to_predict_data.index, future_predictions_gb, label='Gradient Boosting Prediction', linestyle='--', marker='x')

    ax.set_title('Raw Sugar Produced - Original and Predicted Data')
    ax.set_xlabel('Year')
    ax.set_ylabel(target_variable)
    ax.legend()
    
    # Show plot in Streamlit
    st.pyplot(fig)


# Button to perform prediction
if st.button('Predict with Plotly'):
    # Train models with best parameters and make predictions
    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    future_predictions_linear = linear_model.predict(to_predict_data[features])

    # Lasso Regression
    lasso_model = Lasso(alpha=0.001)
    lasso_model.fit(X, y)
    future_predictions_lasso = lasso_model.predict(to_predict_data[features])

    # Random Forest
    rf_model = RandomForestRegressor(max_depth=None, min_samples_split=2, n_estimators=300)
    rf_model.fit(X, y)
    future_predictions_rf = rf_model.predict(to_predict_data[features])

    # Gradient Boosting
    gb_model = GradientBoostingRegressor(learning_rate=0.2, max_depth=3, n_estimators=200)
    gb_model.fit(X, y)
    future_predictions_gb = gb_model.predict(to_predict_data[features])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sugar_cane_data.index, y=y, mode='lines+markers', name='Original Data'))
    fig.add_trace(go.Scatter(x=to_predict_data.index, y=future_predictions_linear, mode='lines+markers', name='Linear Prediction'))
    fig.add_trace(go.Scatter(x=to_predict_data.index, y=future_predictions_lasso, mode='lines+markers', name='Lasso Prediction'))
    fig.add_trace(go.Scatter(x=to_predict_data.index, y=future_predictions_rf, mode='lines+markers', name='Random Forest Prediction'))
    fig.add_trace(go.Scatter(x=to_predict_data.index, y=future_predictions_gb, mode='lines+markers', name='Gradient Boosting Prediction'))

    fig.update_layout(title='Raw Sugar Produced - Original and Predicted Data', xaxis_title='Year', yaxis_title=target_variable)
    st.plotly_chart(fig)
