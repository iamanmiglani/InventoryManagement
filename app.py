import streamlit as st
import pandas as pd
import numpy as np
import random
from prophet import Prophet
import plotly.express as px

# ---------------------------
# Utility Functions with Caching
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data(file_path):
    data = pd.read_csv(file_path)
    # Parse dates assuming format is dd-mm-yyyy; adjust if needed.
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    return data

@st.cache_data(show_spinner=False)
def forecast_demand(data, store_id, product_id, region, periods=7):
    subset = data[(data['StoreID'] == store_id) & 
                  (data['ProductID'] == product_id) & 
                  (data['Region'] == region)]
    if subset.empty:
        st.error("No data available for the selected combination.")
        return None, None
    # Prepare data for Prophet: 'ds' for date, 'y' for UnitsSold
    df_forecast = subset[['Date', 'UnitsSold']].rename(columns={'Date': 'ds', 'UnitsSold': 'y'})
    df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])
    
    model = Prophet(daily_seasonality=True)
    model.fit(df_forecast)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast, df_forecast

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
    st.title("Agentic AI for Retail Inventory Management - V3")
    
    # Load dataset from the 'data' folder
    data_file = "data/inventory_data.csv"
    data = load_data(data_file)
    
    st.sidebar.header("Modules")
    module = st.sidebar.selectbox("Choose a module", 
                                  ["Forecasting", "Inventory Optimization", "Dynamic Pricing", "Reinforcement Learning"])
    
    # Select common filtering parameters for forecast, optimization, and pricing
    store_id = st.sidebar.selectbox("Select Store", sorted(data['StoreID'].unique()))
    product_id = st.sidebar.selectbox("Select Product", sorted(data['ProductID'].unique()))
    region = st.sidebar.selectbox("Select Region", sorted(data['Region'].unique()))
    
    if module == "Forecasting":
        st.header("Time Series Demand Forecasting")
        periods = st.number_input("Forecast Periods (Days)", min_value=1, max_value=30, value=7)
        if st.button("Run Forecast"):
            forecast, hist_data = forecast_demand(data, store_id, product_id, region, periods)
            if forecast is not None:
                st.subheader("Forecast Table")
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))
                
                # Determine the last historical date from the training data
                last_date = hist_data['ds'].max()
                # Label each forecast row as 'Historical' or 'Forecast'
                forecast['Type'] = np.where(forecast['ds'] > last_date, 'Forecast', 'Historical')
                
                st.subheader("Forecast Chart")
                fig = px.line(
                    forecast,
                    x='ds',
                    y='yhat',
                    color='Type',
                    color_discrete_map={"Historical": "blue", "Forecast": "#00FF00"},
                    title='Forecasted Demand (Historical vs. Forecast)',
                    labels={'ds': 'Date', 'yhat': 'Units Sold (Predicted)'}
                )
                
                # Update the forecast trace to use a dashed line style
                for trace in fig.data:
                    if "Forecast" in trace.name:
                        trace.update(line=dict(dash="dash"))
                        
                st.plotly_chart(fig)
    
    elif module == "Inventory Optimization":
        st.header("Inventory Optimization")
        st.write("For the selected store, product, and region, compute reorder quantities based on current inventory and forecasted demand.")
        # Filter data for the selected parameters
        subset = data[(data['StoreID'] == store_id) & 
                      (data['ProductID'] == product_id) & 
                      (data['Region'] == region)]
        if subset.empty:
            st.error("No data available for the selected combination.")
        else:
            # If DemandForecast is not provided, assume it's equal to UnitsSold
            if 'DemandForecast' not in subset.columns:
                subset['DemandForecast'] = subset['UnitsSold']
            subset['ReorderQuantity'] = subset.apply(lambda row: compute_reorder(row['InventoryLevel'], row['DemandForecast']), axis=1)
            
            st.subheader("Inventory Optimization Table")
            st.dataframe(subset[['Date', 'StoreID', 'ProductID', 'Region', 'InventoryLevel', 'DemandForecast', 'ReorderQuantity']].head(10))
            
            st.subheader("Inventory Levels Chart")
            fig = px.line(subset, x='Date', y='InventoryLevel', title='Historical Inventory Levels')
            st.plotly_chart(fig)
    
    elif module == "Dynamic Pricing":
        st.header("Dynamic Pricing")
        st.write("For the selected store, product, and region, dynamic pricing is computed based on inventory vs. forecasted demand.")
        # Filter data for the selected parameters
        subset = data[(data['StoreID'] == store_id) & 
                      (data['ProductID'] == product_id) & 
                      (data['Region'] == region)]
        if subset.empty:
            st.error("No data available for the selected combination.")
        else:
            if 'Price' not in subset.columns:
                st.error("Price column not found in dataset. Please add a 'Price' field to your CSV.")
            else:
                if 'DemandForecast' not in subset.columns:
                    subset['DemandForecast'] = subset['UnitsSold']
                subset['DynamicPrice'] = subset.apply(lambda row: dynamic_price(row['Price'], row['InventoryLevel'], row['DemandForecast']), axis=1)
                st.subheader("Dynamic Pricing Table")
                st.dataframe(subset[['Date', 'StoreID', 'ProductID', 'Region', 'Price', 'DynamicPrice']].head(10))
                
                st.subheader("Dynamic Pricing Chart")
                fig = px.line(subset, x='Date', y='DynamicPrice', title='Dynamic Pricing Over Time')
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
