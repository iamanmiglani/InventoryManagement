import streamlit as st
import pandas as pd
import numpy as np
import random
from prophet import Prophet

# ---------------------------
# Utility Functions with Caching
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data(file_path):
    data = pd.read_csv(file_path)
    # Parse dates (assumed dd-mm-yyyy format); adjust as needed.
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    return data

# Forecasting function that drops the last record from training
@st.cache_data(show_spinner=False)
def forecast_demand(data, store_id, product_id, region, periods=1):
    subset = data[(data['StoreID'] == store_id) & 
                  (data['ProductID'] == product_id) & 
                  (data['Region'] == region)]
    if len(subset) < 2:
        st.error("Not enough data to forecast for the selected combination.")
        return None, None
    # Remove the last record so that the forecast is for that day
    df_train = subset.iloc[:-1]
    df_train_prophet = df_train[['Date', 'UnitsSold']].rename(columns={'Date': 'ds', 'UnitsSold': 'y'})
    df_train_prophet['ds'] = pd.to_datetime(df_train_prophet['ds'])
    
    model = Prophet(daily_seasonality=True)
    model.fit(df_train_prophet)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast, df_train_prophet

# Helper function to extract the forecasted demand (for the next 1 day)
def get_forecasted_demand(data, store_id, product_id, region):
    forecast, _ = forecast_demand(data, store_id, product_id, region, periods=1)
    if forecast is None:
        return None
    # The forecast for the next day is the last row of the forecast DataFrame
    forecasted_demand = forecast.iloc[-1]['yhat']
    return forecasted_demand

# ---------------------------
# Inventory Optimization Module
# ---------------------------
def compute_reorder(inventory_level, demand_forecast, safety_stock=5):
    if inventory_level < demand_forecast:
        return (demand_forecast - inventory_level) + safety_stock
    return 0

# ---------------------------
# Dynamic Pricing Module
# ---------------------------
def dynamic_price(current_price, inventory_level, demand_forecast):
    if demand_forecast <= 0:
        return current_price
    if inventory_level < 0.5 * demand_forecast:
        return current_price * 1.10
    elif inventory_level > 1.5 * demand_forecast:
        return current_price * 0.95
    return current_price

# ---------------------------
# Main Streamlit App
# ---------------------------
def main():
    st.title("Agentic AI for Retail Inventory Management - Backend Optimization")
    
    # Load the dataset from the 'data' folder (make sure the CSV is in the correct location)
    data_file = "data/inventory_data.csv"
    data = load_data(data_file)
    
    st.sidebar.header("Selection Parameters")
    store_id = st.sidebar.selectbox("Select Store", sorted(data['StoreID'].unique()))
    product_id = st.sidebar.selectbox("Select Product", sorted(data['ProductID'].unique()))
    region = st.sidebar.selectbox("Select Region", sorted(data['Region'].unique()))
    
    if st.button("Run Optimization"):
        # Forecast the next day’s demand (computed in the backend)
        forecasted_demand = get_forecasted_demand(data, store_id, product_id, region)
        if forecasted_demand is None:
            st.error("Forecast could not be generated. Check your data.")
            return
        
        # Filter the data for the selected parameters and pick the last available record
        subset = data[(data['StoreID'] == store_id) & 
                      (data['ProductID'] == product_id) & 
                      (data['Region'] == region)]
        if subset.empty:
            st.error("No data available for the selected combination.")
            return
        
        target_row = subset.iloc[-1].copy()
        target_row['ForecastedDemand'] = forecasted_demand
        target_row['ReorderQuantity'] = compute_reorder(target_row['InventoryLevel'], forecasted_demand)
        if 'Price' in target_row:
            target_row['DynamicPrice'] = dynamic_price(target_row['Price'], target_row['InventoryLevel'], forecasted_demand)
        
        # Show final, optimized results for the selected store, product, and region
        st.subheader("Optimized Results for the Forecast Day")
        st.write(target_row[['Date', 'StoreID', 'ProductID', 'Region', 
                               'InventoryLevel', 'ForecastedDemand', 'ReorderQuantity',
                               'Price', 'DynamicPrice']])
        
if __name__ == '__main__':
    main()
