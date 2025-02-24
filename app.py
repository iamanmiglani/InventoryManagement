import streamlit as st
import pandas as pd
import numpy as np
import random
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# Utility Functions with Caching
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data(file_path):
    data = pd.read_csv(file_path)
    # Parse dates assuming dd-mm-yyyy format; adjust as needed.
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    return data

# Forecasting function that drops the last record from training
@st.cache_data(show_spinner=False)
def forecast_demand(data, store_id, product_id, region, periods=1):
    subset = data[(data['StoreID'] == store_id) & 
                  (data['ProductID'] == product_id) & 
                  (data['Region'] == region)]
    if len(subset) < 2:
        st.error("Not enough data to forecast")
        return None, None
    # Remove the last record from training data (simulate unknown UnitsSold for the forecast day)
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
    forecast, hist_data = forecast_demand(data, store_id, product_id, region, periods=1)
    if forecast is None:
        return None
    # The forecast for the next day is the last row in the forecast DataFrame
    forecasted_demand = forecast.iloc[-1]['yhat']
    return forecasted_demand

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
    st.title("Agentic AI for Retail Inventory Management - V4")
    
    # Load dataset from the 'data' folder
    data_file = "data/inventory_data.csv"
    data = load_data(data_file)
    
    st.sidebar.header("Modules")
    module = st.sidebar.selectbox("Choose a module", 
                                  ["Forecasting", "Inventory Optimization", "Dynamic Pricing", "Reinforcement Learning"])
    
    # Common filtering parameters for all modules
    store_id = st.sidebar.selectbox("Select Store", sorted(data['StoreID'].unique()))
    product_id = st.sidebar.selectbox("Select Product", sorted(data['ProductID'].unique()))
    region = st.sidebar.selectbox("Select Region", sorted(data['Region'].unique()))
    
    if module == "Forecasting":
        st.header("Time Series Demand Forecasting (1-Day Forecast)")
        if st.button("Run Forecast"):
            forecast, hist_data = forecast_demand(data, store_id, product_id, region, periods=1)
            if forecast is not None:
                st.subheader("Forecast Table")
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(1))
                
                # Determine the last historical date from training data
                last_date = hist_data['ds'].max()
                # Label rows as 'Historical' or 'Forecast'
                forecast['Type'] = np.where(forecast['ds'] > last_date, 'Forecast', 'Historical')
                
                # Split data for continuous line plotting
                hist_df = forecast[forecast['Type'] == 'Historical']
                fcst_df = forecast[forecast['Type'] == 'Forecast']
                if not fcst_df.empty:
                    # Prepend the last historical point to ensure the lines join seamlessly.
                    last_hist = hist_df.iloc[-1:]
                    fcst_df = pd.concat([last_hist, fcst_df], ignore_index=True)
                
                st.subheader("Forecast Chart")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist_df['ds'],
                    y=hist_df['yhat'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', dash='solid')
                ))
                if not fcst_df.empty:
                    fig.add_trace(go.Scatter(
                        x=fcst_df['ds'],
                        y=fcst_df['yhat'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='#00FF00', dash='dash')
                    ))
                fig.update_layout(title='Forecasted Demand (Historical Joined with Forecast)',
                                  xaxis_title='Date', yaxis_title='Units Sold (Predicted)')
                st.plotly_chart(fig)
    
    elif module == "Inventory Optimization":
        st.header("Inventory Optimization for Forecast Day")
        # Filter data for the selected parameters
        subset = data[(data['StoreID'] == store_id) & 
                      (data['ProductID'] == product_id) & 
                      (data['Region'] == region)]
        if subset.empty:
            st.error("No data available for the selected combination.")
        else:
            # Get forecasted demand for the target (last) day using our helper function
            forecasted_demand = get_forecasted_demand(data, store_id, product_id, region)
            if forecasted_demand is None:
                st.error("Forecast could not be generated.")
            else:
                # Use the last row of the filtered data as the target record
                target_row = subset.iloc[-1].copy()
                target_row['Forecasted Demand'] = forecasted_demand
                target_row['Reorder Quantity'] = compute_reorder(target_row['InventoryLevel'], forecasted_demand)
                
                st.subheader("Optimized Inventory for Forecast Day")
                # Display results vertically (key-value pairs)
                st.markdown(f"**Date:** {target_row['Date']}")
                st.markdown(f"**Store ID:** {target_row['StoreID']}")
                st.markdown(f"**Product ID:** {target_row['ProductID']}")
                st.markdown(f"**Region:** {target_row['Region']}")
                st.markdown(f"**Inventory Level:** {target_row['InventoryLevel']}")
                st.markdown(f"**Forecasted Demand:** {target_row['Forecasted Demand']:.2f}")
                st.markdown(f"**Reorder Quantity:** **{target_row['Reorder Quantity']}**")
    
    elif module == "Dynamic Pricing":
        st.header("Dynamic Pricing for Forecast Day")
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
                forecasted_demand = get_forecasted_demand(data, store_id, product_id, region)
                if forecasted_demand is None:
                    st.error("Forecast could not be generated.")
                else:
                    target_row = subset.iloc[-1].copy()
                    target_row['Forecasted Demand'] = forecasted_demand
                    target_row['Dynamic Price'] = dynamic_price(target_row['Price'], target_row['InventoryLevel'], forecasted_demand)
                    
                    st.subheader("Dynamic Pricing for Forecast Day")
                    st.markdown(f"**Date:** {target_row['Date']}")
                    st.markdown(f"**Store ID:** {target_row['StoreID']}")
                    st.markdown(f"**Product ID:** {target_row['ProductID']}")
                    st.markdown(f"**Region:** {target_row['Region']}")
                    st.markdown(f"**Current Price:** {target_row['Price']}")
                    st.markdown(f"**Forecasted Demand:** {target_row['Forecasted Demand']:.2f}")
                    st.markdown(f"**Dynamic Price:** **{target_row['Dynamic Price']:.2f}**")
    
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
