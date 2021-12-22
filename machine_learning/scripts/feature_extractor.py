from datetime import datetime
import pandas as pd
from pandas.core.arrays import boolean
import numpy as np
import math


class FeatureExtractor:
    """Feature Extractor
    Extracts the autogressive and categorical features of the
    Time Series.

    Args:
        df (pd.DataFrame): Time Series dataframe, with the 'y' column
        representing the observations.
        start_date (datetime.datetime): Represents the beginning of the
        feature extratction. It is important to be careful with the
        start_date and the number of previous days/weeks used during the 
        extraction.

    Attributes:
        features (dict): autoregressive and categorical features.
    """
    def __init__(self, df, start_date):
        self.df = df
        self.start_date = start_date
        self.date_list = None
        self.features = {'date': self.date_list}

    def preprocess(self):
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.date_list = self.df[self.df['date'] >= self.start_date]['date'].tolist()
        self.df['weekday'] = self.df['date'].dt.dayofweek

    @staticmethod
    def get_previous_y(df, n, by_weekday, **kwargs):
        """
        Get the n previous observations, daily or weekly.

        Args:
            df (pd.DataFrame): Time Series dataframe, with 
            the 'y' column representing the observations.
            n (int): number of previous observations.
            by_weekday (bool): If it is True, collects the observations 
            from the exactly same day of week, but from previous weeks. 
            If it is False, collects the n previous day observations.
            date (datetime.datetime) [optional]: Necessary for the
            by_weekday mode, represents the current date of extraction.
        """
        if by_weekday:
            date = kwargs.get('date')
            wd = date.weekday()
            df_weekday = df[df['weekday'] == wd]
            return df_weekday.iloc[-n:].sort_values(
                by='date', ascending=False)['y'].T.values
        else:
            return df.iloc[-n:].sort_values(
                by='date', ascending=False)['y'].T.values   

    @staticmethod
    def get_wom(date: datetime):
        """
        Returns the week of the month of the inputed date.
        """
        wom = math.ceil(date.day/7)
        if wom > 4:
            return 4
        else:
            return wom
 
    def get_autoregressive_features(self, n_days, n_weeks):
        """
        Autoregressive features extraction.

        Args:
            n_days (int): number of previous days to be collected.
            n_weeks (int): number of previous weeks to be collected.
        """
        cols1 = [f'yt-{k}' for k in range(1, n_days+1)]
        cols2 = [f'yt-{k*7}' for k in range(1, n_weeks+1)]
        cols = cols1 + cols2
        autoregressive_data = {}
        for date in self.date_list:
            df_truncate = self.df[self.df['date'] < date].sort_values(by='date')
            values_daily = self.get_previous_y(df_truncate, n_days, False)
            values_weekly = self.get_previous_y(df_truncate, n_weeks, True, date=date)
            values = np.concatenate((values_daily, values_weekly))
            autoregressive_data[date] = values
            
        autoregressive_features = pd.DataFrame(autoregressive_data.values(), columns=cols)
        self.features['autoregressive'] = autoregressive_features

        
    def get_categorical_features(self):
        """
        Categorical features extraction.
        Features returned: day of week, week of month, month of year.
        """
        df_date = pd.DataFrame({'date': self.date_list})
        df_date = df_date.merge(self.df, on='date')[['date', 'weekday']]
        df_date['wom'] = df_date['date'].apply(lambda x: self.get_wom(x))
        df_date['month'] = pd.DatetimeIndex(df_date['date']).month
        df_date = df_date[df_date.columns[1:]]
        
        features = df_date.columns
        categorical_features = pd.DataFrame()
        for feature in features:
            df_dummy = pd.get_dummies(df_date[feature])
            df_dummy = df_dummy.rename(columns={
                col: feature + '_' + str(col) for col in df_dummy.columns
            })
            categorical_features = pd.concat([categorical_features, df_dummy], axis=1)

        self.features['categorical'] = categorical_features
        
    def extract(self, n_days: int, n_weeks: int):
        self.preprocess()
        self.get_autoregressive_features(n_days, n_weeks)
        self.get_categorical_features()