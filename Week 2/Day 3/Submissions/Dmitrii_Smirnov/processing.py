import os

import dask
import dask.dataframe as dd
import pandas as pd

def prepare_data():
    # Choose columns to use
    cols = ['Year', 'Month', 'DayOfWeek', 'Distance', 'DepDelay', 'CRSDepTime', 'UniqueCarrier', 'Origin', 'Dest']
    
    df = dd.read_csv(os.path.join('data', 'nycflights', '*.csv'),
                     usecols=cols,
                     storage_options={'anon': True})
    is_delayed = (df.DepDelay.fillna(16) > 15)
    
    # Remove delay information from training dataframe
    del df['DepDelay']
    
    # Trim all the values in data
    df['CRSDepTime'] = df['CRSDepTime'].clip(upper=2399)
    
    # df: data from which we will learn if flights are delayed
    # is_delayed: whether or not those flights were delayed
    df, is_delayed = dask.persist(df, is_delayed)
    
    # Convert categorical data into numerical
    df_numerical = dd.get_dummies(df.categorize()).persist()
    
    print("- Done")
    
    return df_numerical, is_delayed
                     



