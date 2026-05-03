import pandas as pd
import numpy as np
import gridstatusio
from datetime import datetime, timedelta

client = gridstatusio.GridStatusClient(api_key="1ba2f5619fb54a17bc54408fcf00c4af")

def fetch_all_data(start, end):
    df_lmp = client.get_dataset(
        dataset="ercot_lmp_by_settlement_point",
        start=start, end=end,
        filter_column="location_type", filter_value="Trading Hub"
    )
    df = df_lmp[df_lmp['location'] == 'HB_NORTH'].copy()
    df = df.set_index('interval_start_utc')
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df = df[['lmp']]
    df = df.resample('5min').mean()
    
    #2Day Ahead SPP 
    try:
        dam = client.get_dataset(
            dataset="ercot_spp_day_ahead_hourly",
            start=start, end=end
        )
        dam = dam[dam['location'] == 'HB_NORTH'].copy()
        dam = dam.set_index('interval_start_utc')
        dam.index = pd.to_datetime(dam.index, utc=True)
        dam = dam[['spp']].rename(columns={'spp': 'dam_spp'})
        dam = dam.resample('5min').ffill()
        df = df.join(dam, how='left')
    except Exception as e:
        print(f"DAM fetch failed: {e}")
        df['dam_spp'] = np.nan
    
    # 3 System Load 
    try:
        load = client.get_dataset(
            dataset="ercot_load",
            start=start, end=end
        )
        load.index = pd.to_datetime(load.index, utc=True)
        load = load[['load']].resample('5min').mean()
        df = df.join(load, how='left')
    except Exception as e:
        print(f"Load fetch failed: {e}")
        df['load'] = np.nan
    
    #4 Fuel Mix wind,solar
    try:
        fuel = client.get_dataset(
            dataset="ercot_fuel_mix",
            start=start, end=end
        )
        fuel.index = pd.to_datetime(fuel.index, utc=True)
        fuel = fuel.resample('5min').mean()
        # Sum wind and solar if multiple columns exist
        wind_cols = [c for c in fuel.columns if 'wind' in c.lower()]
        solar_cols = [c for c in fuel.columns if 'solar' in c.lower()]
        if wind_cols:
            df['wind_mw'] = fuel[wind_cols].sum(axis=1)
        if solar_cols:
            df['solar_mw'] = fuel[solar_cols].sum(axis=1)
    except Exception as e:
        print(f"Fuel mix fetch failed: {e}")
        df['wind_mw'] = np.nan
        df['solar_mw'] = np.nan
    
    #Clean up
    df = df.ffill(limit=6).bfill()
    df = df[df['lmp'] < 1000]
    print(f"Final df shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"NaN count:\n{df.isnull().sum()}")
    return df