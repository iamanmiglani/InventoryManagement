import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
from prophet import Prophet
from datetime import datetime
import plotly.graph_objects as go

# ---------------------------
# Configuration and Helper Functions
# ---------------------------
# Define discretization parameters
inv_step = 5
fcst_step = 5
max_inventory = 100
max_fcst = 100

def discretize_value(value, step, max_val):
    return min(max_val, int(round(value / step) * step))

def get_state(inventory, forecast):
    """State: (discretized inventory, discretized forecasted demand)"""
    inv_state = discretize_value(inventory, inv_step, max_inventory)
    fcst_state = discretize_value(forecast, fcst_step, max_fcst)
    return (inv_state, fcst_state)

# Define RL action space: each action is a tuple (order_quantity, price_adjustment_factor)
order_options = [0, 5, 10, 15]
price_adjustments = [1.0, 1.10, 0.95]  # 1.0 = no change, 1.10 = increase 10%, 0.95 = decrease 5%
actions = [(q, p) for q in order_options for p in price_adjustments]

def select_rl_action(Q, state):
    """Select the best action (exploitation only) from the Q-table for the given state.
       If state is unseen, default to (0, 1.0)."""
    if state in Q:
        action_idx = np.argmax(Q[state])
        return actions[action_idx]
    else:
        return (0, 1.0)

def train_forecast_model(df_train):
    model = Prophet(daily_seasonality=True)
    model.fit(df_train)
    return model

def forecast_next_day(model, last_date):
    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)
    # Return forecast for the day immediately after last_date
    forecast_next = forecast[forecast['ds'] > last_date].iloc[0]
    return forecast_next['yhat']

# ---------------------------
# Data Loading
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data(file_path):
    data = pd.read_csv(file_path)
    # Assume Date is in dd-mm-yyyy format; adjust if needed.
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    return data

# Load historical data (Jan 2022 - June 2023) and new datapoints (assume 10 new rows from July 2023 onward)
data = load_data("data/inventory_data.csv")

# For this deployment, we fix the set:
FIXED_STORE_ID = "001"
FIXED_PRODUCT_ID = "101"
FIXED_REGION = "North"

# Split the data
train_end = pd.to_datetime("2023-06-30")
training_data = data[(data['Date'] <= train_end) &
                     (data['StoreID'] == FIXED_STORE_ID) &
                     (data['ProductID'] == FIXED_PRODUCT_ID) &
                     (data['Region'] == FIXED_REGION)].copy()

# Assume new datapoints (10 rows) are after train_end:
new_data = data[(data['Date'] > train_end) &
                (data['StoreID'] == FIXED_STORE_ID) &
                (data['ProductID'] == FIXED_PRODUCT_ID) &
                (data['Region'] == FIXED_REGION)].copy().sort_values("Date")
new_data = new_data.head(10)  # Use only 10 new datapoints

# ---------------------------
# Load pre-trained Q-table (trained offline)
# ---------------------------
with open("trained_q_table.pkl", "rb") as f:
    Q = pickle.load(f)

# ---------------------------
# Streamlit App UI
# ---------------------------
st.title("Integrated RL-Driven Retail Inventory & Pricing Decisions")
st.markdown("**Fixed Set:** StoreID: 001, ProductID: 101, Region: North")

# Let user select which new datapoint (day) to simulate
day_index = st.slider("Select Simulation Day (1 to 10)", min_value=1, max_value=10, value=1)
sim_row = new_data.iloc[day_index - 1]

# For simulation, we'll assume:
# - The current actual inventory and price come from the last available historical record.
current_inventory = training_data.iloc[-1]['InventoryLevel']
current_price = training_data.iloc[-1]['Price']

# Combine training data with new data up to the day before the current simulation day:
available_data = pd.concat([training_data, new_data[new_data['Date'] < sim_row['Date']]])
available_data = available_data.sort_values("Date")

# Prepare Prophet training data (using all available data)
df_train_prophet = available_data[['Date', 'UnitsSold']].rename(columns={'Date': 'ds', 'UnitsSold': 'y'})
df_train_prophet['ds'] = pd.to_datetime(df_train_prophet['ds'])

# Train the forecasting model
with st.spinner("Training forecasting model on available data..."):
    try:
        forecast_model = train_forecast_model(df_train_prophet)
    except Exception as e:
        st.error(f"Forecast model training failed: {e}")
        st.stop()

last_date_train = df_train_prophet['ds'].max()
forecasted_demand = forecast_next_day(forecast_model, last_date_train)

# Form current state using RL: state = (discretized inventory, discretized forecasted demand)
state = get_state(current_inventory, forecasted_demand)

# Use the pre-trained Q-table to select the RL action (order quantity and price adjustment)
order_qty, price_adj = select_rl_action(Q, state)

# Simulate operational update:
# 1. New inventory = current inventory + order_qty
new_inventory = current_inventory + order_qty
# 2. Adjusted price = current price * price_adj
new_price = current_price * price_adj

# For the simulation day, we have an actual demand (from new_data)
actual_demand = sim_row['UnitsSold']

# Sales: minimum of available inventory (new_inventory) and actual demand
sales = min(new_inventory, actual_demand)
ending_inventory = new_inventory - sales

# For reward computation (for display purposes), compute revenue and costs:
revenue = sales * new_price
ordering_cost = order_qty * 5  # using same cost structure as before
holding_cost = ending_inventory * 0.1
reward = revenue - ordering_cost - holding_cost

# ---------------------------
# Display Results
# ---------------------------
st.header("Simulation Results for Selected Day")
st.markdown(f"**Simulation Date:** {sim_row['Date'].date()}")
st.markdown(f"**Forecasted Demand for Next Day:** {forecasted_demand:.2f}")
st.markdown(f"**Current Inventory:** {current_inventory}")
st.markdown(f"**RL Recommended Order Quantity:** {order_qty}")
st.markdown(f"**RL Recommended Price Adjustment Factor:** {price_adj}")
st.markdown(f"**New Price (after adjustment):** {new_price:.2f}")
st.markdown(f"**Actual Demand on Simulation Day:** {actual_demand}")
st.markdown(f"**Sales:** {sales}")
st.markdown(f"**Ending Inventory:** {ending_inventory}")
st.markdown(f"**Revenue:** {revenue:.2f}")
st.markdown(f"**Ordering Cost:** {ordering_cost}")
st.markdown(f"**Holding Cost:** {holding_cost:.2f}")
st.markdown(f"**Reward:** **{reward:.2f}**")

st.markdown("---")
st.markdown("This simulation demonstrates how the RL agent—trained offline and loaded as a Q-table—integrates with forecasting, inventory optimization, and dynamic pricing. For each new day, the system uses updated historical data to forecast demand and then uses the RL policy to decide on ordering and pricing adjustments in real time.")
