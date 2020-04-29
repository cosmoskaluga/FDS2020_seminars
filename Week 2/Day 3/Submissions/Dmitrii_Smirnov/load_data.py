from __future__ import print_function

import os
import numpy as np
import pandas as pd
import tarfile
import urllib.request
import zipfile
from glob import glob
from config import params


def flights():
    
    data_dir = params.get('data_dir', None)
    nycflights_gz = params.get('flights_raw', None)
    nycflights = params.get('flightdir', None)
    flightjson = params.get('jsondir', None)
    url = params.get('url', None)
    nrows = params.get('nrows', None)
    
    flights_raw = os.path.join(data_dir, nycflights_gz)
    flightdir = os.path.join(data_dir, nycflights)
    jsondir = os.path.join(data_dir, flightjson)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(flights_raw):
        print("- Downloading NYC Flights dataset... ", end='', flush=True)
        urllib.request.urlretrieve(url, flights_raw)
        print("done", flush=True)

    if not os.path.exists(flightdir):
        print("- Extracting flight data... ", end='', flush=True)
        with tarfile.open(flights_raw, mode='r:gz') as flights:
            flights.extractall(data_dir +'/')
        print("done", flush=True)

    if not os.path.exists(jsondir):
        print("- Creating json data... ", end='', flush=True)
        os.mkdir(jsondir)
        for path in glob(os.path.join(data_dir, nycflights, '*.csv')):
            prefix = os.path.splitext(os.path.basename(path))[0]
            df = pd.read_csv(path).iloc[:nrows]
            df.to_json(os.path.join(data_dir, flightjson, prefix + '.json'),
                       orient='records', lines=True)
        print("done", flush=True)



