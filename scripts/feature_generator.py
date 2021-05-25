'''
File name: sensor_values_prep.py
Author: 930233
Date created: 01/04/2021
Date last modified: 24/05/2013
Python Version: 3.7
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import tensorflow as tf
import pickle

from datetime import datetime, timedelta
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

from scripts.errors import CreatingTestWithoutScalerError, DayOutGreaterThanDayIn

# Class to perform feature engineering
class FeatureGenerator():
    def __init__(self, df, sensor_df, df_cols, features, labels, win_size, day_pred, df_date_offset, max_date=None, wash=False):
        if (win_size < day_pred):
            raise DayOutGreaterThanDayIn()
        
        self.df_cols = df_cols
        self.features = features
        self.labels = labels
        self.wash = wash
        self.df_date_offset = df_date_offset
        self.day_pred = day_pred
        
        self.max_date = datetime.strptime(max_date, "%Y-%m-%d").date() if max_date is not None else max_date
        self.df = self.format_df(df)
        self.min_date = self.max_date - timedelta(self.df_date_offset)
        
        self.sensor_df = sensor_df
        self.win_size = win_size
        self.scaler = None
        
    def __repr__(self):
        return "\n".join([
            f"Dataset Length: {len(self.df)}",
            f"WC: {self.df_cols['wc']}",
            f"Window Size: {self.win_size}",
            f"Date Range: {self.min_date} - {self.max_date}",
            f"Features: {self.features}",
            f"Days to predict: {self.day_pred}"
        ])

    # Merge sensor df with CIM_TA_EC data
    # Only keep columns with less tha (1-min_data) NaN rows
    def format_df(self, df):
        wc = self.df_cols["wc"]
        wc_date = self.df_cols["wc_date"]
        c = self.df_cols["count"]
        f = self.df_cols["fail"]
        
        wc_df = df[self.df_cols.values()]
        wc_df = wc_df[~wc_df[wc].isna()] # keep rows where wc != nan
        
        if (self.wash):
            wc_df.loc[:, wc] = wc_df[wc].str.slice(0, 5) # remove lane if wc == Wash
            
        wc_df.loc[:, wc_date] = pd.to_datetime(wc_df[wc_date]).dt.date # remove time from date
        
        if (self.max_date is None):
            self.max_date = wc[wc_date].max()
        
        min_day = self.max_date - timedelta(self.df_date_offset) # find min_date by offsetting from max_date
        
        wc_df = wc_df[(wc_df[wc_date] >= min_day) & (wc_df[wc_date] <= self.max_date)]
        wc_df = wc_df.sort_values(by=[wc_date], ascending=False).reset_index(drop=True) # sort by date
        
        wc_grp = wc_df.groupby([wc, wc_date])[[c, f]].sum().reset_index() # groupby date & wc
        
        return wc_grp

    # Transform dataset into pivot table
    def gen_table(self, min_data=0.9, interpolate=True):
        wc = self.df_cols["wc"]
        wc_date = self.df_cols["wc_date"]
        c = self.df_cols["count"]
        f = self.df_cols["fail"]
        
        if (self.sensor_df is None):
            wc_merge = self.df.copy()
        else:
            self.sensor_df.loc[:, "dte"] = pd.to_datetime(self.sensor_df["dte"]).dt.date
            wc_merge = pd.merge(self.df, self.sensor_df, left_on=[wc_date, wc], right_on=["dte", "station"], how="left")
        
        wc_merge["fail_percentage"] = wc_merge[f].div(wc_merge[c])
        wc_table = pd.pivot_table(wc_merge, index=[wc_date], columns=[wc]) # convert to pivot table
        wc_table = wc_table[self.features] # only keep feature cols
        min_data = len(wc_table) * min_data
        wc_table = wc_table.loc[:, wc_table.count() >= min_data] # only keep wc with data >= min_data
        
        # Find common wcs and keep them
        level_0 = list(set(wc_table.columns.get_level_values(0)))
        common_cols = set(wc_table[level_0[0]].columns)
        
        for i in level_0[1:]:
            common_cols = common_cols.intersection(set(wc_table[i].columns))
        common_cols = list(common_cols)
        
        keep_cols = []
        
        for i in wc_table.columns:
            if i[1] in common_cols:
                keep_cols.append(i)
                
        wc_table = wc_table[keep_cols]
        
        if (interpolate):
            wc_table = wc_table.interpolate(limit_direction="both")
        else:
            wc_table = wc_table.fillna(0)
        
        self.wc_cols = common_cols
        self.df = wc_table

    # Train test split
    def df_split(self, train_size=0.8):
        if (train_size == 0 and self.scaler is None):
            raise CreatingTestWithoutScalerError()
        
        df_all = {}
        
        # Rearranging pivot table
        for wc in self.wc_cols:
            temp_list = []
            v_cols = list(self.df.columns.get_level_values(0).unique().tolist())
            for v in v_cols:
                temp_v = self.df[v].loc[:, wc].values
                temp_list.append(temp_v)
            temp_list = list(zip(*temp_list))
            temp_df = pd.DataFrame(temp_list, columns=v_cols)
            
            df_all[wc] = temp_df
            
        self.df_all = df_all.copy()
        
        if (train_size == 1):
            self.rolling_normalize(df_all, is_train=True)
        elif (train_size == 0):
            self.rolling_normalize(df_all, is_train=False)
        else:
            s = pd.Series(df_all)
            train_df, test_df = [i.to_dict() for i in train_test_split(s, train_size=train_size)]
            self.rolling_normalize(train_df, is_train=True)
            self.rolling_normalize(test_df, is_train=False)

    # Apply rolling mean and min max scaling to df
    def rolling_normalize(self, df, is_train):
        # Rolling mean for dataset
        for i in df.keys():
            df[i] = df[i].rolling(window=self.win_size, min_periods=1).mean()
        
        # MinMax scaling
        if (is_train):
            self.unscaled_train = np.array(df)
            x = []
            for wc in df.keys():
                x.extend(df[wc].values)
            if (not self.scaler):
                scaler = MinMaxScaler()
                scaler = scaler.fit(x)
                self.scaler = scaler
        else:
            self.unscaled_test = np.array(df)
            
        scaled_df = {}
        
        for i in df.keys():
            scaled_df[i] = self.scaler.transform(df[i])
        
        if (is_train):
            self.train_df = scaled_df
        else:
            self.test_df = scaled_df

    # Save min max scaler
    def save_scaler(self, path):
        pickle.dump(self.scaler, open(path, "wb"))

    # Load min max scaler
    def load_scaler(self, path):
        self.scaler = pickle.load(open(path, "rb"))

    # Split features and labels in data window
    def split_window(self, features):
        inputs = features[:-1] # taking all windows from [:-1]
        labels = features[1:, :self.day_pred, -1] # taking fail_percent from [1:]
        
        return inputs, labels

    # Apply data windowing to datasets
    # For train datasets, data windowing is applyed to individual WCs first, then they are joined together.
    # We do this instead of combining all WCs then applying data windowing (easier) 
    # because we do want an instance where the features of the last few rows of WC1 is 
    # used to predict the label of the first row of WC2 
    def make_dataset(self, is_train):
        features = []
        labels = []
        
        if (is_train):
            combined = []
        else:
            combined = {}
        ds = None
        
        if (is_train):
            df = self.train_df.copy()
        else:
            df = self.test_df.copy()
        
        for i in df.keys():
            # Apply data windowing
            ds = timeseries_dataset_from_array(data=df[i], targets=None, sequence_length=self.win_size)
            ds = ds.map(self.split_window)
            
            if (is_train):
                for j in ds.__iter__():
                    features.extend(j[0].numpy())
                    labels.extend(j[1].numpy())
            else:
                for j in ds.__iter__():
                    features= j[0].numpy()
                    labels = j[1].numpy()
                combined[i] = [features, labels]
            if (is_train):
                combined = [np.array(features), np.array(labels)]
                
        return combined

    # Generate train and test datasets
    @property
    def train(self):
        df = self.make_dataset(is_train=True)
        return df

    @property
    def test(self):
        df = self.make_dataset(is_train=False)
        return df
    