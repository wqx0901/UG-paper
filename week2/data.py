import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 当前文件所在目录的上级目录
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class GEFCom2014():
    def __init__(self, test_ratio=0.33):
        # read data
        self.df = pd.read_csv(os.path.join(_BASE_DIR, 'data', 'GEFCom2014', 'L1-train.csv'))
        # clean data
        self.df.drop(columns=['TIMESTAMP','ZONEID'], inplace=True)
        self.df.dropna(subset=['LOAD'], inplace=True)
        self.df = self.df.head(1000)
        # normalize data
        self.scaler = MinMaxScaler()
        self.df = self.scaler.fit_transform(self.df)  # numpy array
        # split dataset
        self.n_samples = len(self.df)
        self.n_train = int(self.n_samples * (1 - test_ratio))
        self.n_test = int(self.n_samples * test_ratio)
        self.df_train = self.df[:self.n_train]
        self.df_test = self.df[self.n_train:]

    def get_slided_dataset(self, d_num, h_num, is_train=True):
        df = self.df_train if is_train else self.df_test
        X_list, Y_list = [], []
        for t in range(0, len(df) - h_num - d_num + 1):
            x = df[t : t + d_num, 0]          # (d_num, 1)
            y = df[t + d_num : t + d_num + h_num, 0]  # (h_num, 1)
            X_list.append(x)
            Y_list.append(y)
        return np.array(X_list), np.array(Y_list) # (n_samples, d_num, 1), (n_samples, h_num, 1)

if __name__ == "__main__":
    # test
    gefcom2014 = GEFCom2014()
    print('Get train dataset (hum = 24)')
    X, Y = gefcom2014.get_slided_dataset(d_num=24, h_num=24, is_train=True)
    print(X.shape, Y.shape)
    print('Get train dataset (hum = 1)')
    X, Y = gefcom2014.get_slided_dataset(d_num=24, h_num=1, is_train=True)
    print(X.shape, Y.shape)
    print('Get test dataset (hum = 24)')
    X, Y = gefcom2014.get_slided_dataset(d_num=24, h_num=24, is_train=False)
    print(X.shape, Y.shape)