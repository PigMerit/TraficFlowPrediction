"""
Processing the data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def process_data(train, test, lags):
    """Process data
    Reshape and split train\test data.

    # Arguments
        train: String, name of .csv train file.
        test: String, name of .csv test file.
        lags: integer, time lag.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        X_test: ndarray.
        y_test: ndarray.
        scaler: StandardScaler.
    """
    attr = 'Lane 1 Flow (Veh/5 Minutes)'
    df1 = pd.read_csv(train, encoding='utf-8').fillna(0)
    df2 = pd.read_csv(test, encoding='utf-8').fillna(0)

    print("*********************读进来的原始数据************************************")
    print("train")
    print(df1)
    print("test")
    print(df2)
    # scaler = StandardScaler().fit(df1[attr].values)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
    flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(df2[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    print("*********************修改一下的原始数据************************************")
    print("train")
    print(len(flow1))
    print(flow1)
    print("test")
    print(len(flow2))
    print(flow2)
    train, test = [], []
    for i in range(lags, len(flow1)):
        train.append(flow1[i - lags: i + 1])
    for i in range(lags, len(flow2)):
        test.append(flow2[i - lags: i + 1])
    print("*********************循环一下的原始数据************************************")
    print("train")
    print(len(train))
    print(train[0])
    print("test")
    print(len(test))
    print(test[0])
    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)
    print("*********************np一下的原始数据************************************")
    print("train")
    print(train)
    print("test")
    print(test)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]
    # print("*********************最终返回的数据************************************")
    # print("train")
    # print(train)
    # print("test")
    # print(test)
    return X_train, y_train, X_test, y_test, scaler
