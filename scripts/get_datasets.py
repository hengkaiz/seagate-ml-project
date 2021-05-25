'''
File name: sensor_values_prep.py
Author: 930233
Date created: 01/04/2021
Date last modified: 21/05/2013
Python Version: 3.7
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time

import credentials
import jaydebeapi

from scripts.sensor_sql_query import sql_query

from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Download and transform datasets for machine learning
class GetDatasets():
    def __init__(self, wc, min_date, max_date, sensors=None, weather=None, train=False, sensor_file_name=None, notifications=True):
        self.wc = wc
        if (sensors is None):
            self.sensors = ["di_recirc", "mst1", "mst2", "cs_osr"]
        if (weather is None):
            self.weather = ["no2_mean"]
        if (sensor_file_name is None):
            self.sensor_file_name = "_wash_flow_values_"
        if (train):
            self.sensor_file_name += "train"
        else:
            self.sensor_file_name += "test"
        self.min_date = datetime.strptime(min_date, '%Y-%m-%d')
        self.max_date = datetime.strptime(max_date, '%Y-%m-%d')
        self.n = notifications
        self.sensor_dfs = {}
        self.train = train
            
    def __repr__(self):
        return "\n".join([
            f"WC: {self.wc}",
            f"Date Range: {self.min_date} to {self.max_date}",
            f"Sensors: {self.sensors}",
            f"Weather: {self.weather}"
        ])

    # Download weather dataset from data.gov
    def download_weather_data(self):
        diff = (self.max_date - self.min_date).days
        location = "north"
        curr_date = self.min_date
        env_list = []
        
        if (self.n):
            print(f"\nDownloading weather dataset")
        
        for i in range(diff):
            params = {"date" : datetime.strftime(curr_date, '%Y-%m-%d')}
            response = requests.get(url='https://api.data.gov.sg/v1/environment/psi', params=params)
            temp_json = response.json()
            temp_items = temp_json["items"]

            curr_date = curr_date + timedelta(days=1)
            o3 = co = no2 = so2 = 0
            no_hrs = len(temp_items)

            if (no_hrs == 0):
                continue

            for i in temp_items:
                temp_readings = i["readings"]
                no2 += temp_readings["no2_one_hour_max"][location]

            no2 /= no_hrs
            env_list.append([curr_date, no2])
            time.sleep(0.5)

        env_df = pd.DataFrame(env_list, columns=["dte", "no2_mean"])
        
        if (self.n):
            print(f"Weather data downloaded")
        
        self.env_df = env_df

    # Download wash sensor values from hadoop
    def download_sensor_data(self):
        if (self.n):
            print("\nDownloading sensor values")
        conn = jaydebeapi.connect(r"com.facebook.presto.jdbc.PrestoDriver", 
            f"jdbc:presto://mediadatalakepresto.seagate.com:8449/hive/default?SSL=true&SSLKeyStorePath={credentials.PRESTO_FOLDER}keystore.jks&SSLKeyStorePassword=Seagate@123",
            [str(credentials.USER), str(credentials.PASS)],
            f"{credentials.PRESTO_FOLDER}presto-jdbc-334.jar")

        curs = conn.cursor()
        min_date = str(self.min_date.date())
        max_date = str(self.max_date.date())

        for wc in self.wc:
            sql = sql_query(min_date=min_date, max_date=max_date, wc=(wc)) # SQL query for wash sensor values
            curs.execute(sql)
            data = curs.fetchall()
            data = pd.DataFrame(data, columns=["di_recirc_date_tmst", "mst1_date_tmst", "mst2_date_tmst", "cs_osr_date_tmst", "di_recirc_flow", "mst1_flow", "mst2_flow", "cs_osr_flow"])
            data.to_csv(f"datasets\{wc.lower()}{self.sensor_file_name}.csv", index=False)

            if (self.n):
                print(f"{wc} flow values downloaded")
        
        curs.close()
        conn.close()

        if (self.n):
            print("All sensor values downloaded")

    def split_df(self, df, date_col, flow_col):
        temp_df = df[[date_col, flow_col]].copy()
        temp_df.loc[:, date_col] = pd.to_datetime(temp_df[date_col])
        temp_df = temp_df.dropna()
        
        return temp_df

    # Use isolation forest to get anomaly count
    def get_anomalies(self, df):
        # Standard scaling
        scaler = StandardScaler()
        v = df.iloc[:, -1].values
        v = v.reshape(-1, 1)
        np_scaled = scaler.fit_transform(v)
        data = pd.DataFrame(np_scaled)
        
        # Isolation Forest
        model = IsolationForest(contamination=0.01, n_jobs=-2)
        model.fit(data)
        
        # Predict anomalies
        anomaly = pd.Series(model.predict(data))
        
        temp_df = df.copy()
        
        temp_df.loc[:, "standardized"] = data.values
        temp_df.loc[:, "anomaly"] = anomaly.values
        
        return temp_df

    # Group df by day
    def grp_by_day(self, df, date_col, flow_col, sensor):
        temp_df = df.copy()
        temp_df.loc[:, "dte"] = df[date_col].apply(lambda x : x.date())
        
        a_grp = temp_df.groupby(["dte"])["anomaly"].sum() # group by anomalies sum
        ms_grp = temp_df.groupby(["dte"])["standardized"].mean() # group by standardized mean
        mf_grp = temp_df.groupby(["dte"])[flow_col].mean() # group by flow mean
        
        df_merge = pd.merge(a_grp, ms_grp, on=["dte"])
        df_merge = pd.merge(df_merge, mf_grp, on=["dte"])
        
        df_merge = df_merge.rename(columns={"anomaly" : sensor + "_anomaly_sum",
                                            "standardized" : sensor + "_standardized_mean",
                                            flow_col : flow_col + "_mean"})
        
        return df_merge

    # Transform downloaded sensor df
    def prep_sensor_data(self):
        csv_location="datasets/"
        csv_format = f"{self.sensor_file_name}.csv"
                
        self.sensor_dfs = {}
        if (self.n):
            print()
            
        for s in self.wc:
            df = pd.read_csv(f"{csv_location}{s.lower()}{csv_format}")

            sensor_dfs = {}
            for i in self.sensors:
                date_col = i + "_date_tmst"
                flow_col = i + "_flow"
                sensor_dfs[i] = self.split_df(df, date_col, flow_col)
            if (self.n):
                print(f"Detecting anomalies for {s}")
                
            anomaly_dfs = {}
            for i in self.sensors:
                anomaly_dfs[i] = self.get_anomalies(sensor_dfs[i])
                sensor_name = anomaly_dfs[i].columns[1]
            if (self.n):
                print(f"Grouping {s}")
                
            grp_dfs = {}
            for i in self.sensors:
                date_col = i + "_date_tmst"
                flow_col = i + "_flow"
                grp_dfs[i] = self.grp_by_day(anomaly_dfs[i], date_col, flow_col, i)
                sensor_name = grp_dfs[i].columns[-1]
            merge_df = pd.DataFrame()
            merge_df = pd.merge(grp_dfs[self.sensors[0]], grp_dfs[self.sensors[1]], on=["dte"], how="outer")
            for i in self.sensors[2:]:
                merge_df = pd.merge(merge_df, grp_dfs[i], on=["dte"], how="outer")
            self.sensor_dfs[s] = merge_df.reset_index()

    # Combine different wc df into 1 df
    def combine_wc(self):
        combined_df = pd.DataFrame()
        for s in self.wc:
            temp_df = self.sensor_dfs[s]
            temp_df.loc[:, "station"] = s
            combined_df = combined_df.append(temp_df)
        
        self.merge_sensor_df = combined_df
        self.merge_sensor_df["dte"] = pd.to_datetime(self.merge_sensor_df["dte"])
        
        if (self.n):
            print("\nSensor dfs merged")

    # Add weather data to combined df
    def add_weather(self):
        env_df = self.env_df
        env_stacked = pd.DataFrame()

        for i in self.wc:
            temp_df = env_df.copy()
            temp_df["station"] = i
            env_stacked = env_stacked.append(temp_df)
            
        env_stacked["dte"] = pd.to_datetime(env_stacked["dte"])
        self.df = pd.merge(self.merge_sensor_df, env_stacked, on=["dte", "station"], how="outer")
        
        if (self.n):
            print("\nWeather data added to sensor df")

    # Main function to run
    def run(self):
        self.download_sensor_data()
        self.download_weather_data()
        self.prep_sensor_data()
        self.combine_wc()
        self.add_weather()