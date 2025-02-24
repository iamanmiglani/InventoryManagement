import streamlit as st
import pandas as pd
import numpy as np
import random
from prophet import Prophet
import plotly.express as px
import datetime

# ---------------------------
# Utility Functions with Caching
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

@st.cache_data(show_spinner=False)
def forecast_demand(data, store_id, product_id, region, periods=7):
    subset = data[(data['StoreID'] == store_id) & 
                  (data['ProductID'] == product_id) & 
                  (data['Region'] == region)]
    if subset.empty:
        st.error("No data available for the selected combination.")
        return None
    df_forecast = subset[['Date', 'UnitsSold']].rename(columns={'Date': 'ds', 'UnitsSold': 'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(df_forecast)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

# ---------------------------
# Module 2: Inventory Optimization
# ---------------------------
def compute_reorder(inventory_level, demand_forecast, safety_stock=5):
    if inventory_level < demand_forecast:
        return (demand_forecast - inventory_level) + safety_stock
    return 0

# ---------------------------
# Module 3: Dynamic Pricing
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
# Module 5: Reinforcement Learning (RL) Simulation
# ---------------------------
@st.cache_data(show_spinner=False)
def run_rl_simulation(episodes=1000, days_per_episode=30, demand_mean=20):
    max_inventory = 100
    inventory_levels = list(range(0, max_inventory + 1, 5))
    actions = [0, 5, 10, 15]
    
    def simulate_day(inventory, order_qty):
        inventory += order_qty  # New stock arrives
        demand = np.random.poisson(demand_mean)
        sales = min(inventory, demand)
        new_inventory = inventory - sales
        revenue = sales * 10      # Sale price per unit
        order_cost = order_qty * 5  # Ordering cost per unit
        holding_cost = new_inventory * 0.1  # Holding cost per unit
        reward = revenue - order_cost - holding_cost
        return new_inventory, reward
    
    num_states = len(inventory_levels)
    num_actions = len(actions)
    Q = np.zeros((num_states, num_actions))
    alpha = 0.1    # Learning rate
    gamma = 0.95   # Discount factor
    epsilon = 0.1  # Exploration rate
    
    def get_state_index(inventory):
        inv = min(max_inventory, int(round(inventory / 5) * 5))
        return inventory_levels.index(inv)
    
    for episode in range(episodes):
        inventory = 50  # Reset starting inventory each episode
        for day in range(days_per_episode):
            state_idx = get_state_index(inventory)
            if random.random() < epsilon:
                action_idx = random.choice(range(num_actions))
            else:
                action_idx = np.argmax(Q[state_idx])
            order_qty = actions[action_idx]
            new_inventory, reward = simulate_day(inventory, order_qty)
            new_state_idx = get_state_index(new_inventory)
            best_next_action = np.max(Q[new_state_idx])
            Q[state_idx, action_idx] += alpha * (reward + gamma * best_next_action - Q[state_idx, action_idx])
            inventory = new_inventory
    return Q

# ---------------------------
# Main Streamlit App
# ---------------------------
def main():
    st.title("Agentic AI for Retail Inventory Management - V2")
    
    # Load the dataset from the 'data' folder in your repository
    data_file = "data/inventory_data.csv"
    data = load_data(data_file)
    
    st.sidebar.header("Modules")
    module = st.sidebar.selectbox("Choose a module", 
                                  ["Forecasting", "Inventory Optimization", "Dynamic Pricing", "Reinforcement Learning"])
    
    if module == "Forecasting":
        st.header("Time Series Demand Forecasting")
        store_id = st.selectbox("Select Store", sorted(data['StoreID'].unique()))
        product_id = st.selectbox("Select Product", sorted(data['ProductID'].unique()))
        region = st.selectbox("Select Region", sorted(data['Region'].unique()))
        periods = st.number_input("Forecast Periods (Days)", min_value=1, max_value=30, value=7)
        if st.button("Run Forecast"):
            forecast = forecast_demand(data, store_id, product_id, region, periods)
            if forecast is not None:
                st.write("Forecasted Demand:")
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))
                fig = px.line(forecast, x='ds', y='yhat', title='Forecasted Demand')
                st.plotly_chart(fig)
    
    elif module == "Inventory Optimization":
        st.header("Inventory Optimization")
        st.write("Computed Reorder Quantities Based on Current Inventory vs. Forecasted Demand")
        if 'DemandForecast' not in data.columns:
            data['DemandForecast'] = data['UnitsSold']
        data['ReorderQuantity'] = data.apply(lambda row: compute_reorder(row['InventoryLevel'], row['DemandForecast']), axis=1)
        st.dataframe(data[['StoreID', 'ProductID', 'Region', 'InventoryLevel', 'DemandForecast', 'ReorderQuantity']].head(10))
    
    elif module == "Dynamic Pricing":
        st.header("Dynamic Pricing")
        st.write("Dynamic Price Adjustments Based on Inventory Levels")
        if 'Price' not in data.columns:
            st.error("Price column not found in dataset. Please add a 'Price' field to your CSV.")
        else:
            if 'DemandForecast' not in data.columns:
                data['DemandForecast'] = data['UnitsSold']
            data['DynamicPrice'] = data.apply(lambda row: dynamic_price(row['Price'], row['InventoryLevel'], row['DemandForecast']), axis=1)
            st.dataframe(data[['StoreID', 'ProductID', 'Region', 'Price', 'DynamicPrice']].head(10))
            fig = px.line(data, x='Date', y='DynamicPrice', title='Dynamic Pricing Over Time')
            st.plotly_chart(fig)
    
    elif module == "Reinforcement Learning":
        st.header("Reinforcement Learning Simulation")
        episodes = st.slider("Number of Episodes", min_value=100, max_value=2000, value=1000, step=100)
        days_per_episode = st.slider("Days per Episode", min_value=10, max_value=60, value=30, step=5)
        demand_mean = st.number_input("Average Daily Demand", min_value=1, max_value=100, value=20)
        Q = run_rl_simulation(episodes=episodes, days_per_episode=days_per_episode, demand_mean=demand_mean)
        st.write("Trained Q-Table:")
        q_df = pd.DataFrame(Q, columns=["Order 0", "Order 5", "Order 10", "Order 15"])
        st.dataframe(q_df)

if __name__ == '__main__':
    main()
