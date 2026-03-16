"""
Electricity 数据集加载器
支持多用户数据拼接（Global）用于联合训练
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Electricity:
    """Electricity 数据集：多用户用电负荷，支持 Global 拼接训练"""

    def __init__(self, data_path=None, test_ratio=0.2, max_users=None, max_samples_per_user=None):
        """
        Args:
            data_path: CSV 路径，默认 data/electricity/electricity.csv
            test_ratio: 测试集比例
            max_users: 最多加载的用户数（用于快速实验），None 表示全部
            max_samples_per_user: 每用户最多样本数，None 表示全部
        """
        path = data_path or os.path.join(_BASE_DIR, 'data', 'electricity', 'electricity.csv')
        self.df = pd.read_csv(path)
        # 去掉 date 和 OT 列，保留用户列 (0, 1, 2, ...)
        user_cols = [c for c in self.df.columns if c not in ('date', 'OT') and str(c).isdigit()]
        self.user_cols = user_cols[:max_users] if max_users else user_cols
        self.n_users = len(self.user_cols)
        self.test_ratio = test_ratio
        self.max_samples_per_user = max_samples_per_user
        self.scalers = {}  # 每用户独立 scaler，或全局 scaler

    def get_global_slided_dataset(self, d_num=24, h_num=1, use_global_scaler=True):
        """
        Global 模式：将多个用户的数据拼接后构造滑窗数据集
        每个用户单独做滑窗，再 concat 所有用户的 (X, y) 样本

        Args:
            d_num: 回看窗口长度（小时）
            h_num: 预测步长（小时）
            use_global_scaler: True 用全局 MinMax，False 用每用户独立 scaler

        Returns:
            X_train, y_train, X_test, y_test: numpy arrays
        """
        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []

        if use_global_scaler:
            # 先收集所有数据做全局 fit
            all_values = []
            for col in self.user_cols:
                vals = self.df[col].dropna().values.astype(np.float32)
                all_values.append(vals)
            all_values = np.concatenate(all_values)
            global_scaler = MinMaxScaler()
            global_scaler.fit(all_values.reshape(-1, 1))

        for col in self.user_cols:
            vals = self.df[col].dropna().values.astype(np.float32)
            if self.max_samples_per_user:
                vals = vals[:self.max_samples_per_user]
            n = len(vals)
            n_test = int(n * self.test_ratio)
            n_train = n - n_test

            train_vals = vals[:n_train]
            test_vals = vals[n_train:]

            if use_global_scaler:
                train_norm = global_scaler.transform(train_vals.reshape(-1, 1)).flatten()
                test_norm = global_scaler.transform(test_vals.reshape(-1, 1)).flatten()
            else:
                scaler = MinMaxScaler()
                train_norm = scaler.fit_transform(train_vals.reshape(-1, 1)).flatten()
                test_norm = scaler.transform(test_vals.reshape(-1, 1)).flatten()
                self.scalers[col] = scaler

            for t in range(0, len(train_norm) - d_num - h_num + 1):
                x = train_norm[t : t + d_num]
                y = train_norm[t + d_num : t + d_num + h_num]
                X_train_list.append(x)
                y_train_list.append(y)
            for t in range(0, len(test_norm) - d_num - h_num + 1):
                x = test_norm[t : t + d_num]
                y = test_norm[t + d_num : t + d_num + h_num]
                X_test_list.append(x)
                y_test_list.append(y)

        X_train = np.array(X_train_list, dtype=np.float32)
        y_train = np.array(y_train_list, dtype=np.float32)
        X_test = np.array(X_test_list, dtype=np.float32)
        y_test = np.array(y_test_list, dtype=np.float32)

        self.global_scaler = global_scaler if use_global_scaler else None
        self.scaler = self.global_scaler
        return X_train, y_train, X_test, y_test
