from sklearn.utils import all_estimators
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense


def get_LSTM(n_features, y_dim):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_features, y_dim)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
            
    return model

def create_models(n_features, y_dim):
    # Scikit-learn models
    estimators = all_estimators(type_filter='regressor')

    all_regs = {}
    for name, RegressorClass in estimators:
        try:
            print('Appending', name)
            reg = RegressorClass()
            all_regs[name] = reg
        except Exception as e:
            print(e)

    all_regs.update({
        XGBRegressor.__name__: XGBRegressor(),
        LGBMRegressor.__name__: LGBMRegressor(),
        'LSTM': get_LSTM(n_features, y_dim)
    })

    return all_regs