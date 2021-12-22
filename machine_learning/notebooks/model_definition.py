from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


INITIAL_CONFIG = {
    'input_shape': 13
}


def build_model(initial_config: dict):
    """Builds the DNN regression model
    """
    input_shape = INITIAL_CONFIG['input_shape']
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[input_shape]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


models = [
    LinearRegression(),
    RandomForestRegressor(random_state=42),
    build_model(INITIAL_CONFIG),
]

MODEL_DICT = {type(model).__name__: model for model in models}