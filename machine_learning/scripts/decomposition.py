import numpy as np
import pandas as pd


class TimeSeriesDecomposer:
    """Time Series Decomposer
    Extracts the Trend, Seasonality and Remainder components of the
    Time Series.

    Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting: 
    principles and practice, 2nd edition, OTexts: Melbourne, 
    Australia. OTexts.com/fpp2. Accessed on 12-20-2021.
    Available on https://otexts.com/fpp2/classical-decomposition.html

    Args:
        df (pd.DataFrame): Time Series dataframe, with the 'y' column
        representing the observations.

    Attributes:
        df (pd.DataFrame): Time Series dataframe.
    """

    def __init__(self, df):
        self.df = df

    @staticmethod
    def compute_ma(df:pd.DataFrame, m:int):
        k = int((m - 1) / 2)
        ma = []
        for i in range(len(df)):
            i_start = i - k
            i_end = i + k + 1
            if i_start < 0:
                ma.append(np.nan)
            else:
                ma.append(df.iloc[i_start:i_end]['y'].mean())

        df[f'ma_{m}'] = ma
        return df

    def additive_decomposition(self, m:int, 
        season:int):
        df = self.compute_ma(self.df, m)
        df['trend'] = df[f'ma_{m}']
        df['detrend_y'] = df['y'] - df['trend']
        df['seasonality'] = df['detrend_y'].rolling(window=season).mean()
        df['remainder'] = df['y'] - df['trend'] - df['seasonality']
        df['at'] = df['trend'] + df['remainder']
        df['st'] = df['seasonality']

        return df

    def classical(self, m, season, method:str):
        if method == 'additive':
            return self.additive_decomposition(m, season)

    def decompose(self, m:int, season:int, 
        decomposition_type:str, **kwargs):
        if decomposition_type == 'classical':
            classical_method = kwargs.get('classical_method')
            return self.classical(m, season, classical_method)
