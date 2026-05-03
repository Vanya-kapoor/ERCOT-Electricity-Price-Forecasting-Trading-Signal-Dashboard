import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta
from ercot_features import make_features
from ercot_data import fetch_all_data

st.set_page_config(page_title='ERCOT LMP Predictor', layout='wide')
st.title('ERCOT HB_NORTH LMP Predictor')

@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model('ercot_xgb_model_wf.json')
    return model

model = load_model()
#i will get latest data
@st.cache_data(ttl=300)
def get_data():
    end = datetime.now()
    start = (end - timedelta(days=3)).strftime("%Y-%m-%d")
    end = end.strftime("%Y-%m-%d")
    return fetch_all_data(start, end)

df = get_data()
df_feat = make_features(df)


print(f"df shape after fetch: {df.shape}")
print(f"df_feat shape after make_features: {df_feat.shape}")

if df_feat.empty:
    st.error("No data after feature engineering. Check that external datasets have overlapping dates with LMP data.")
    st.stop()

feature_cols = [c for c in df_feat.columns if c not in ['target', 'lmp']]
model_features = model.get_booster().feature_names

#last 24 hours
hist_features = df_feat[model_features].iloc[-288:]
hist_preds = model.predict(hist_features)

current_price = df_feat['lmp'].iloc[-1]
next_price = model.predict(df_feat[model_features].iloc[-1:])[0]
delta = next_price - current_price
#this is shown on the streamlit app
col1, col2, col3 = st.columns(3)
col1.metric("Current LMP", f"${current_price:.2f}")
col2.metric("Predicted Next", f"${next_price:.2f}")
col3.metric("Expected Move", f"${delta:+.2f}")

if delta > 2:
    st.success("BUY")
elif delta < -2:
    st.error("SELL")
else:
    st.info("HOLD")


# if 'dam_spp' in df_feat.columns:
#     dam_price = df_feat['dam_spp'].iloc[-1]
#     st.write(f"Day Ahead: ${dam_price:.2f} | Spread: ${current_price - dam_price:+.2f}")

# if 'load' in df_feat.columns:
#     load_val = df_feat['load'].iloc[-1]
#     st.write(f"System Load: {load_val:,.0f} MW")

# if 'wind_mw' in df_feat.columns and 'solar_mw' in df_feat.columns:
#     wind = df_feat['wind_mw'].iloc[-1]
#     solar = df_feat['solar_mw'].iloc[-1]
#     st.write(f"Wind: {wind:,.0f} MW | Solar: {solar:,.0f} MW")


fig = go.Figure()
fig.add_trace(go.Scatter(x=df_feat.index[-288:], y=df_feat['lmp'].iloc[-288:], name='Actual', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df_feat.index[-288:] + timedelta(minutes=5), y=hist_preds, name='Predicted', line=dict(color='red', dash='dash')))
fig.update_layout(title='Last 24 Hours', xaxis_title='Time', yaxis_title='$/MWh', height=450)
st.plotly_chart(fig, use_container_width=True)

if st.button('Refresh'):
    st.cache_data.clear()
    st.rerun()