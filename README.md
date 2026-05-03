# ERCOT-Electricity-Price-Forecasting-Trading-Signal-Dashboard
Real-time electricity price prediction system for ERCOT HB_NORTH using XGBoost. The project takes real time and day time market, system load, and renewable generation data from gridstatus.io to forecast short-term LMP movements and generate trading signals through an interactive Streamlit dashboard.

This is how the streamlit dashboard looks like 
<img width="1914" height="969" alt="image" src="https://github.com/user-attachments/assets/a0b90e67-d449-4110-abfd-702711e0a01b" />

##  What This Project Does

- Fetches real-time ERCOT electricity market data using the GridStatus API  
- Builds time-series features including lag values, rolling statistics, and demand/supply signals  
- Trains an XGBoost regression model to predict next-step LMP  
- Generates BUY / SELL / HOLD trading signals based on predicted price movement  
- Displays a live dashboard using Streamlit with interactive Plotly visualizations  

---

##  Features Engineered

### Time Features
- Hour of day  
- Day of week  
- Month  
- Peak-hour indicators  

###  Lag Features
- Historical LMP values (1, 12, 288 timesteps)

### Rolling Statistics
- Rolling mean  
- Rolling standard deviation  
- Rolling maximum (short & long windows)

### Market Features
- Real-time vs day-ahead price spread  

###  Load Features
- System demand (load level)  
- Load changes and ratios  

###  Renewable Features
- Wind generation  
- Solar generation  

### System Feature
- Net load = Demand − (Wind + Solar)

---

##  Model

- **Algorithm:** XGBoost Regressor  
- **Target:** Next timestep Real-Time LMP  
- **Input:** Engineered time-series + market fundamentals  
- **Output:** Short-term electricity price forecast  

---

##  Trading Logic

- If predicted price > current price + threshold → **BUY**  
- If predicted price < current price − threshold → **SELL**  
- Otherwise → **HOLD**

---

##  Dashboard (Streamlit)

The interactive dashboard includes:

- Current LMP  
- Predicted next price  
- Expected price movement  
- Trading signal (BUY / SELL / HOLD)  
- System load, wind, and solar snapshot  
- 24-hour actual vs predicted price chart  

---

##  Key Challenges Solved

- Time-series alignment across multiple data sources  
- Handling missing load and renewable generation values  
- Rolling-window feature engineering without data leakage  
- Real-time API caching for performance optimization  
- Ensuring consistency between training and inference features  

---

##  Tech Stack

- Python  
- Pandas, NumPy  
- XGBoost  
- Streamlit  
- Plotly  
- GridStatus API  

---

##  Key Insight

This project demonstrates how electricity price forecasting requires combining:
- Market signals (LMP, DAM prices)
- Physical system variables (load, wind, solar)
- Temporal dependencies (lags + rolling patterns)
