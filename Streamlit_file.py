# ---------------------------------------
# 📦 Imports
# ---------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
from prophet import Prophet
from datetime import datetime
import plotly.graph_objects as go

# ---------------------------------------
# ⚙️ Configuration and Helper Functions
# ---------------------------------------

# Discretization steps and limits for state representation
inv_step = 5
fcst_step = 5
max_inventory = 100
max_fcst = 100

def discretize_value(value, step, max_val):
    """Round the value to the nearest step, capped at max_val."""
    return min(max_val, int(round(value / step) * step))

def get_state(inventory, forecast):
    """Get discrete state tuple from current inventory and forecast."""
    inv_state = discretize_value(inventory, inv_step, max_inventory)
    fcst_state = discretize_value(forecast, fcst_step, max_fcst)
    return (inv_state, fcst_state)

# Define action space for RL: combinations of order quantity and price adjustment
order_options = [0, 5, 10, 15]                # Possible quantities to order
price_adjustments = [1.0, 1.10, 0.95]         # Price changes: no change, +10%, -5%
actions = [(q, p) for q in order_options for p in price_adjustments]

def select_rl_action(Q, state):
    """
    Given a state, return the best action from the Q-table.
    If unseen state, return default action (0 order, no price change).
    """
    if state in Q:
        action_idx = np.argmax(Q[state])
        return actions[action_idx]
    else:
        return (0, 1.0)

def train_forecast_model(df_train):
    """Train Prophet model on historical sales data."""
    model = Prophet(daily_seasonality=True)
    model.fit(df_train)
    return model

def forecast_next_day(model, last_date):
    """Forecast demand for the next day after last known date."""
    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)
    forecast_next = forecast[forecast['ds'] > last_date].iloc[0]
    return forecast_next['yhat']

# ---------------------------------------
# 📥 Data Loading
# ---------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file_path):
    """Load data from CSV and convert Date column to datetime."""
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    return data

# ---------------------------------------
# 🧠 Load Pre-trained Q-table
# ---------------------------------------
with open("trained_q_table.pkl", "rb") as f:
    Q = pickle.load(f)

# ---------------------------------------
# 🖼️ Streamlit App UI (V7.1)
# ---------------------------------------
st.title("Integrated RL-Driven Retail Inventory & Pricing Decisions (V7.1)")
st.markdown("**Fixed Set:** StoreID: S001, ProductID: P0015, Region: East")

# Load and filter data
data = load_data("data/inventory_data.csv")

# Split data into training (before Dec 1, 2023) and simulation (after)
train_end = pd.to_datetime("2023-11-30")
training_data = data[(data['Date'] <= train_end) &
                     (data['StoreID'] == "S001") &
                     (data['ProductID'] == "P0015") &
                     (data['Region'] == "East")].copy()

new_data = data[(data['Date'] > train_end) &
                (data['StoreID'] == "S001") &
                (data['ProductID'] == "P0015") &
                (data['Region'] == "East")].copy().sort_values("Date")

# Handle empty simulation set
if new_data.empty:
    st.error("No new datapoints available for simulation (dates after November 30, 2023).")
    st.stop()

# Let user select a simulation date
sim_dates = sorted(new_data['Date'].dt.date.unique().tolist())
selected_date = st.selectbox("Select Simulation Date", sim_dates)

# Get data row for selected simulation date
sim_row = new_data[new_data['Date'].dt.date == selected_date].iloc[0]

# Get last known inventory and price from training data
current_inventory = training_data.iloc[-1]['InventoryLevel']
current_price = training_data.iloc[-1]['Price']

st.markdown(f"**Starting Inventory:** {current_inventory}")
st.markdown(f"**Starting Price:** {current_price}")

# Merge training data with any new data before selected simulation date
available_data = pd.concat([training_data, new_data[new_data['Date'] < pd.to_datetime(selected_date)]])
available_data = available_data.sort_values("Date")

# Prepare training set for Prophet
df_train_prophet = available_data[['Date', 'UnitsSold']].rename(columns={'Date': 'ds', 'UnitsSold': 'y'})
df_train_prophet['ds'] = pd.to_datetime(df_train_prophet['ds'])

# Train Prophet model on available data
with st.spinner("Training forecasting model on available data..."):
    try:
        forecast_model = train_forecast_model(df_train_prophet)
    except Exception as e:
        st.error(f"Forecast model training failed: {e}")
        st.stop()

# Forecast demand for simulation date
last_date_train = df_train_prophet['ds'].max()
forecasted_demand = forecast_next_day(forecast_model, last_date_train)

# Form RL state from current inventory and forecast
state = get_state(current_inventory, forecasted_demand)

# Select action (order quantity + price adjustment)
order_qty, price_adj = select_rl_action(Q, state)

# Apply action: update inventory and price
new_inventory = current_inventory + order_qty
new_price = current_price * price_adj

# Get actual demand on simulation day
actual_demand = sim_row['UnitsSold']

# Compute business KPIs
sales = min(new_inventory, actual_demand)
ending_inventory = new_inventory - sales
revenue = sales * new_price
ordering_cost = order_qty * 5               # Assume ₹5/unit ordering cost
holding_cost = ending_inventory * 0.1       # ₹0.10 per unit holding cost
reward = revenue - ordering_cost - holding_cost

# Forecasting accuracy metrics
forecast_error = abs(actual_demand - forecasted_demand)
forecast_percentage_error = (forecast_error / actual_demand * 100) if actual_demand != 0 else 0

# ---------------------------------------
# 📊 Display Simulation Results
# ---------------------------------------
st.header("Simulation Results for Selected Day")
st.markdown(f"**Simulation Date:** {selected_date}")
st.markdown(f"**Forecasted Demand for Next Day:** {forecasted_demand:.2f}")
st.markdown(f"**Forecast Error:** {forecast_error:.2f} units ({forecast_percentage_error:.2f}%)")
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

# Footer Note
st.markdown("---")
st.markdown("This simulation integrates the RL layer (using a pre-trained Q-table) with forecasting, inventory optimization, and dynamic pricing. It uses new datapoints (selected by date from those after November 30, 2023) to test forecast accuracy and operational performance in a real-time–like setting.")
