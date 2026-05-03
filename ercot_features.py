import numpy as np
def make_features(df):
    df = df.copy()
    
   
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['is_peak_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)
    
  
    for lag in [1, 2, 3, 6, 12, 24, 288]:
        df[f'lag_lmp_{lag}'] = df['lmp'].shift(lag)

    df['lmprollmean12'] = df['lmp'].rolling(12).mean()
    df['lmprollmean288'] = df['lmp'].rolling(288).mean()
    df['lmprollstd12'] = df['lmp'].rolling(12).std()
    df['lmprollmax12'] = df['lmp'].rolling(12).max()  # <-- ADD THIS BACK
    
    #Day Ahead spread
    if 'dam_spp' in df.columns:
        df['rt_dam_spread'] = df['lmp'] - df['dam_spp']
        df['dam_lag1'] = df['dam_spp'].shift(1)
        df['dam_ratio'] = df['lmp'] / df['dam_spp'].replace(0, np.nan)
    
    #Load features
    if 'load' in df.columns:
        df['load_lag1'] = df['load'].shift(1)
        df['load_rollmean12'] = df['load'].rolling(12).mean()
        df['load_change'] = df['load'].diff(1)
        df['load_ratio'] = df['load'] / df['load_rollmean12']
    
    #Renewable features
    if 'wind_mw' in df.columns:
        df['wind_lag1'] = df['wind_mw'].shift(1)
        df['wind_ratio'] = df['wind_mw'] / df['load'].replace(0, np.nan)
    
    if 'solar_mw' in df.columns:
        df['solar_lag1'] = df['solar_mw'].shift(1)
        df['solar_ratio'] = df['solar_mw'] / df['load'].replace(0, np.nan)
    
    #Net load
    if all(c in df.columns for c in ['load', 'wind_mw', 'solar_mw']):
        df['net_load'] = df['load'] - df['wind_mw'] - df['solar_mw']
        df['net_load_ratio'] = df['net_load'] / df['load'].replace(0, np.nan)
    
    #Price changes
    df['lmpdelta1'] = df['lmp'].diff(1)
    df['lmpdelta2'] = df['lmp'].diff(12)
    
    #Target
    df['target'] = df['lmp'].shift(-1)
    
    #Clean
    df = df.dropna(subset=['target', 'lmp', 'lag_lmp_1'])

    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
    return df