params = {
    "data_dir": "data",
    "flights_raw": "nycflights.tar.gz",
    "flightdir": "nycflights",
    "jsondir": "flightjson",
    "url": "https://storage.googleapis.com/dask-tutorial-data/nycflights.tar.gz",
    "nrows": 10000
}


param_grid_svs = {
    'C': [0.001, 10.0],
        'kernel': ['rbf'],
}

param_grid_xgboost = {'objective': 'binary:logistic',
                        'nround': 1000,
                        'max_depth': 16,
                        'eta': 0.01,
                        'subsample': 0.5,
                        'min_child_weight': 1,
                        'tree_method': 'hist',
                        'grow_policy': 'lossguide'
}
