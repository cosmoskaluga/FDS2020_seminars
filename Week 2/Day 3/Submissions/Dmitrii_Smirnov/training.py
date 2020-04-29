import numpy as np
import pandas as pd
import time
import dask
import joblib
import xgboost as xgb
from dask.distributed import Client, progress
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import dask_xgboost as dxgb
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import metrics



def sklearn_regressor(data_train, data_test, labels_train, labels_test):
    print("\n\n***** Sklearn regressor *****")
    estimator = SVC(gamma='auto', random_state=0, probability=True)
    
    # Due to the lack of RAM I will use only 2000 rows from test and training data
    train_rows = 2000
    test_rows = 400
    
    X_train, X_test = data_train.head(train_rows).values, data_test.head(test_rows).values
    y_train, y_test = labels_train.head(train_rows).values, labels_test.head(test_rows).values
    
    from config import param_grid_svs
    
    start = time.time()
    grid_search = GridSearchCV(estimator, param_grid_svs, verbose=2, cv=2, n_jobs=-1)

    with joblib.parallel_backend('dask'):
        grid_search.fit(X_train, y_train)
        
        print(grid_search.best_params_)
        
        grid_search_time = str(time.time() - start)
        
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        
        print(" ")
        print("Detailed classification report:")
        y_true, y_pred = y_test, grid_search.predict(X_test)
        print(classification_report(y_true, y_pred))
        accuracy = metrics.accuracy_score(y_true, y_pred)
        print("Accuracy:", accuracy)
        total_time_svc = str(time.time() - start)
        print("- Done")

        return [grid_search_time, total_time_svc, accuracy]



def daskml_regressor(client, data_train, data_test, labels_train, labels_test):
    print("\n\n***** Dask ml XGBoost *****")
    start = time.time()
    
    from config import param_grid_xgboost
    
    bst = dxgb.train(client, param_grid_xgboost, data_train, labels_train)
    pdxgb_train_time =  str(time.time() - start)

    predictions = dxgb.predict(client, bst, data_test).persist()

    accuracy = roc_auc_score(labels_test.compute(), predictions.compute())
    print("Accuracy:", accuracy)
    print("- Done")

    return [0, pdxgb_train_time, accuracy]


def train_models(df, is_delayed):
    data_train, data_test = df.random_split([0.8, 0.2], random_state=1234)
    labels_train, labels_test = is_delayed.random_split([0.8, 0.2], random_state=1234)
    
    # Initialyze the cluster
    client = Client()

    # Start training with sklearn SVC regressor
    skl_output = sklearn_regressor(data_train, data_test, labels_train, labels_test)
    # And then launch XGBoost with Dask
    xgboost_output = daskml_regressor(client, data_train, data_test, labels_train, labels_test)


    df_output = pd.DataFrame({'SklearnSVC': skl_output,
                  'DaskXGBoost': xgboost_output })
                  
    df_output.to_csv('output.csv', index=['GridSearchTime', 'TrainTime', 'Accuracy'])


    print("- Done")





    


