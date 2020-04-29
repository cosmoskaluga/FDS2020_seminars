#from __future__ import print_function

import numpy as np
import pandas as pd

import load_data
import processing
import training


def main():
    print("1. Setting up data directory")
    load_data.flights()
    print("2. Preparing data")
    df, is_delayed = processing.prepare_data()
    print("3. Making prediction")
    training.train_models(df, is_delayed)


if __name__ == '__main__':
    main()
