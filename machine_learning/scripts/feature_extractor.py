import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from math import ceil
from functools import reduce


class FeatureExtractor:
    def __init__(self, df):
        self.df = df
        self.df['date'] = pd.to_datetime(self.df['date'])
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('FeatureExtractor')
        self.logger.setLevel(logging.INFO)

    def get_price_features(self):
        self.logger.info('Extracting price features')
        selected_cols = ['date','item_nbr', 'store_nbr', 'price']
        grid_df = self.df[selected_cols]

        grid_df['price_max'] = grid_df.groupby(['store_nbr','item_nbr'])['price'].transform('max')
        grid_df['price_min'] = grid_df.groupby(['store_nbr','item_nbr'])['price'].transform('min')
        grid_df['price_std'] = grid_df.groupby(['store_nbr','item_nbr'])['price'].transform('std')
        grid_df['price_mean'] = grid_df.groupby(['store_nbr','item_nbr'])['price'].transform('mean')


        grid_df['price_norm'] = grid_df['price']/grid_df['price_max']

        grid_df['price_nunique'] = grid_df.groupby(['store_nbr','item_nbr'])['price'].transform('nunique') 
        grid_df['item_nunique'] = grid_df.groupby(['store_nbr','price'])['item_nbr'].transform('nunique')

        grid_df['month'] = grid_df['date'].dt.month.astype(np.int8)
        grid_df['year'] = grid_df['date'].dt.year

        grid_df['price_momentum'] = grid_df['price']/grid_df.groupby(['store_nbr','item_nbr'])['price'].transform(lambda x: x.shift(1))
        grid_df['price_momentum_m'] = grid_df['price']/grid_df.groupby(['store_nbr','item_nbr','month'])['price'].transform('mean')
        grid_df['price_momentum_y'] = grid_df['price']/grid_df.groupby(['store_nbr','item_nbr','year'])['price'].transform('mean')
        
        grid_df.drop(columns=['month', 'year'], inplace=True)
        
        return grid_df

    def get_date_features(self):
        self.logger.info('Extracting date features')
        selected_cols = ['date', 'item_nbr', 'store_nbr']
        grid_df = self.df[selected_cols]

        grid_df['day'] = grid_df['date'].dt.day.astype(np.int8)
        grid_df['week'] = grid_df['date'].dt.week.astype(np.int8)
        grid_df['month'] = grid_df['date'].dt.month.astype(np.int8)
        grid_df['year'] = grid_df['date'].dt.year
        grid_df['year'] = (grid_df['year'] - grid_df['year'].min()).astype(np.int8)
        grid_df['wom'] = grid_df['day'].apply(lambda x: ceil(x/7)).astype(np.int8)
        grid_df['dow'] = grid_df['date'].dt.dayofweek.astype(np.int8) 
        grid_df['weekend'] = (grid_df['dow']>=5).astype(np.int8)
        
        return grid_df

    def get_lagged_features(self):
        self.logger.info('Extracting lagged features')
        TARGET = 'unit_sales'
        SHIFT_DAY = 28
        selected_cols = [
            'date', 'item_nbr', 'store_nbr', TARGET
        ]
        grid_df = self.df[selected_cols]

        LAG_DAYS = [col for col in range(SHIFT_DAY,SHIFT_DAY+15)]
        grid_df = grid_df.assign(**{
                '{}_lag_{}'.format(col, l): grid_df.groupby(
                    ['item_nbr', 'store_nbr']
                )[col].transform(lambda x: x.shift(l))
                for l in LAG_DAYS
                for col in [TARGET]
            })

        for col in list(grid_df):
            if 'lag' in col:
                grid_df[col] = grid_df[col].astype(np.float16)

        for i in [7,14,30,60,180]:
            grid_df['rolling_mean_'+str(i)] = grid_df.groupby(
                ['item_nbr', 'store_nbr']
            )[TARGET].transform(lambda x: x.shift(SHIFT_DAY).rolling(i).mean()).astype(np.float16)
            
            grid_df['rolling_std_'+str(i)]  = grid_df.groupby(
                ['item_nbr', 'store_nbr']
            )[TARGET].transform(lambda x: x.shift(SHIFT_DAY).rolling(i).std()).astype(np.float16)

        for d_shift in [1,7,14]:
            for d_window in [7,14,30,60]:
                col_name = 'rolling_mean_shift_'+str(d_shift)+'_'+str(d_window)
                grid_df[col_name] = grid_df.groupby(
                    ['item_nbr', 'store_nbr']
                )[TARGET].transform(lambda x: x.shift(d_shift).rolling(d_window).mean()).astype(np.float16)
        
        return grid_df

    def extract(self):
        categorical_features = self.get_date_features()
        price_features = self.get_price_features()
        lagged_features = self.get_lagged_features()
        
        num_features = price_features.merge(
            lagged_features, on=['date', 'item_nbr', 'store_nbr']
        )

        self.logger.info('Features extracted successfully')

        return {
            'categorical': categorical_features,
            'numeric': num_features
        }