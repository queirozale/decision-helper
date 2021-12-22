import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

def eval_model(model, samples):
    result = {}
    for feature_name, feature_samples in samples.items():
        train_data = feature_samples['train']
        X_train, y_train = train_data['X']['data'], train_data['y']['data']

        test_data = feature_samples['test']
        X_test, y_test = test_data['X']['data'], test_data['y']['data']

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_scaler = train_data['y']['scaler']
        test_scaler = test_data['y']['scaler']

        y_true_train = train_scaler.inverse_transform(y_train)
        y_true_test = train_scaler.inverse_transform(y_test)

        y_pred_train = train_scaler.inverse_transform(y_pred_train.reshape(-1, 1))
        y_pred_test = train_scaler.inverse_transform(y_pred_test.reshape(-1, 1))

        result[feature_name] = {
            'train': {
                'true': y_true_train,
                'pred': y_pred_train
            },
            'test': {
                'true': y_true_test,
                'pred': y_pred_test
            }
        }

    y_true_train = result['autoregressive']['train']['true'] + result['categorical']['train']['true']
    y_pred_train = result['autoregressive']['train']['pred'] + result['categorical']['train']['pred']

    y_true_test = result['autoregressive']['test']['true'] + result['categorical']['test']['true']
    y_pred_test = result['autoregressive']['test']['pred'] + result['categorical']['test']['pred']

    results_metrics = {
        'mae/avg': {
            'train': mean_absolute_error(y_true_train, y_pred_train)/np.mean(y_true_train),
            'test': mean_absolute_error(y_true_test, y_pred_test)/np.mean(y_true_test)
        }
    }
    
    return results_metrics