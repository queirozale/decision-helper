import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_samples(features, df_, train_size):
    sample_keys = [
        ('autoregressive', 'at'),
        ('categorical', 'st')
    ]
    
    samples = {}
    for sample_key in sample_keys:
        feature_name, y_true_name = sample_key
        feature_samples = {}
        data = {
            'train': {
                'X': features[feature_name][:train_size],
                'y': df_[[y_true_name]][:train_size]
            },
            'test': {
                'X': features[feature_name][train_size:],
                'y': df_[[y_true_name]][train_size:]
            }
        }

        for set_ in data.keys():
            feature_samples[set_] = {}
            for k, vec in data[set_].items():
                scaler = MinMaxScaler()
                vec_scaled = scaler.fit_transform(vec)
                feature_samples[set_][k] = {
                    'data': vec_scaled,
                    'scaler': scaler
                }

        samples[feature_name] = feature_samples
        
    return samples