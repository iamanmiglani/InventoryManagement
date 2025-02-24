import streamlit as st
import pandas as pd
import numpy as np
import random
from prophet import Prophet
import plotly.express as px
import datetime

# ---------------------------
# Module 1: Time Series Demand Forecasting
# ---------------------------
def forecast_demand(data, store_id, product_id, region, periods=7):
    subset = data[(data['StoreID'] == store_id) & (data['ProductID'] == product_id) & (data['Region'] == region)]
    if subset.empty:
        st.write("No data available for the selected combination.")
        return None
    # Prepare data for Prophet (columns 'ds' and 'y')
    df_forecast = subset[['Date', 'UnitsSold']].rename(columns={'Date': 'ds', 'UnitsSold': 'y'})
    df_forecast['ds'] = pd.to_datetime(df_forecast['ds'], dayfirst=True)
    
    model = Prophet()
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
    else:
        return 0

# ---------------------------
# Module 3: Dynamic Pricing
# ---------------------------
def dynamic_price(current_price, inventory_level, demand_forecast):
    if inventory_level < 0.5 * demand_forecast:
        return current_price * 1.10
    elif inventory_level > 1.5 * demand_forecast:
        return current_price * 0.95
    else:
        return current_price

# ---------------------------
# Module 5: Reinforcement Learning (Q-Learning Simulation)
# ---------------------------
def run_rl_simulation(episodes=1000, days_per_episode=30):
    max_inventory = 100
    inventory_levels = list(range(0, max_inventory + 1, 5))
    actions = [0, 5, 10, 15]
    demand_mean = 20
    
    def simulate_day(inventory, order_qty, demand_mean=demand_mean):
        inventory += order_qty
        demand = np.random.poisson(demand_mean)
        sales = min(inventory, demand)
        new_inventory = inventory - sales
        revenue = sales * 10      # Sale price per unit
        order_cost = order_qty * 5  # Cost per unit ordered
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
        inventory = 50  # Starting inventory
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
            Q[state_idx, action_idx] = Q[state_idx, action_idx] + alpha * (reward + gamma * best_next_action - Q[state_idx, action_idx])
            inventory = new_inventory
    return Q

# ---------------------------
# Main Streamlit App
# ---------------------------
def main():
    st.title("Agentic AI for Retail Inventory Management")
    
    # Load the dataset from the 'data' folder in the repository
    data_file = "data/inventory_data.csv"
    data = pd.read_csv(data_file)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    
    # Sidebar navigation for selecting modules
    st.sidebar.title("Modules")
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
        # For demonstration, if DemandForecast is not in the dataset, assume it equals UnitsSold
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
        st.write("Simulating Q-learning for Inventory Ordering Decisions")
        episodes = st.slider("Number of Episodes", min_value=100, max_value=2000, value=1000, step=100)
        Q = run_rl_simulation(episodes=episodes)
        st.write("Trained Q-Table:")
        q_df = pd.DataFrame(Q, columns=["Order 0", "Order 5", "Order 10", "Order 15"])
        st.dataframe(q_df)

if __name__ == '__main__':
    main()
