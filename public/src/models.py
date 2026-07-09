"""
Model configurations and factory functions.

This module provides standardized model configurations used in the paper,
including Linear Regression, XGBoost, LightGBM, MLP, and LSTM.
"""

import numpy as np
from typing import Dict, Callable, Optional, List
from sklearn.linear_model import Ridge


def get_model_configs(
    include_deep_learning: bool = False,
) -> Dict[str, Callable]:
    """
    Get dictionary of model factory functions.

    Parameters
    ----------
    include_deep_learning : bool
        Whether to include MLP and LSTM models

    Returns
    -------
    model_configs : dict
        Dictionary mapping model names to factory functions
    """
    import xgboost as xgb
    import lightgbm as lgb

    configs = {
        'LinearRegression': lambda: Ridge(alpha=1.0),

        'XGBoost': lambda: xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
        ),

        'LightGBM': lambda: lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        ),
    }

    if include_deep_learning:
        configs['MLP'] = lambda: create_mlp_model(input_dim=None)
        configs['LSTM'] = lambda: create_lstm_model(input_dim=None)

    return configs


def create_mlp_model(
    input_dim: Optional[int],
    hidden_dims: List[int] = [128, 64, 32],
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
):
    """
    Create a Multi-Layer Perceptron model for regression.

    Parameters
    ----------
    input_dim : int
        Number of input features
    hidden_dims : list
        Hidden layer dimensions
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Adam optimizer learning rate

    Returns
    -------
    model : keras.Sequential
        Compiled MLP model
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("TensorFlow is required for deep learning models. Install with: pip install tensorflow")

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.BatchNormalization(),
    ])

    for dim in hidden_dims:
        model.add(layers.Dense(dim, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.BatchNormalization())

    model.add(layers.Dense(1, activation='linear'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'],
    )

    return model


def create_lstm_model(
    input_dim: Optional[int],
    lstm_units: List[int] = [64, 32],
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
):
    """
    Create an LSTM model for regression.

    NOTE: This implementation uses timesteps=1, meaning the LSTM processes
    each observation independently without temporal context. This is effectively
    a dense network with LSTM cell computations rather than true sequential
    modeling. For proper temporal modeling, modify to use 12+ timesteps with
    appropriate data windowing in the KerasModelWrapper.

    Parameters
    ----------
    input_dim : int
        Number of input features
    lstm_units : list
        LSTM layer units
    dropout_rate : float
        Dropout rate
    learning_rate : float
        Adam optimizer learning rate

    Returns
    -------
    model : keras.Sequential
        Compiled LSTM model
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("TensorFlow is required for deep learning models. Install with: pip install tensorflow")

    model = keras.Sequential([
        layers.Input(shape=(1, input_dim)),  # (timesteps=1, features)
    ])

    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)
        model.add(layers.LSTM(units, return_sequences=return_sequences, dropout=dropout_rate))

    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='linear'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'],
    )

    return model


class KerasModelWrapper:
    """
    Wrapper to give Keras models a scikit-learn compatible interface.

    This allows using Keras models in the same walk-forward framework
    as scikit-learn models.
    """

    def __init__(
        self,
        model_type: str = 'MLP',
        hidden_dims: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 30,
        batch_size: int = 512,
        validation_split: float = 0.2,
        early_stopping_patience: int = 5,
        verbose: int = 0,
    ):
        self.model_type = model_type
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        self.model = None
        self.scaler = None

    def fit(self, X, y):
        """Fit the model."""
        from tensorflow.keras import callbacks
        from sklearn.preprocessing import StandardScaler

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Create model
        input_dim = X.shape[1]
        if self.model_type == 'MLP':
            self.model = create_mlp_model(
                input_dim, self.hidden_dims, self.dropout_rate, self.learning_rate
            )
            X_input = X_scaled
        else:  # LSTM
            self.model = create_lstm_model(
                input_dim, self.hidden_dims[:2], self.dropout_rate, self.learning_rate
            )
            X_input = X_scaled.reshape(-1, 1, input_dim)

        # Early stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
        )

        # Train
        self.model.fit(
            X_input, y,
            validation_split=self.validation_split,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=self.verbose,
        )

        return self

    def predict(self, X):
        """Generate predictions."""
        X_scaled = self.scaler.transform(X)

        if self.model_type == 'LSTM':
            X_scaled = X_scaled.reshape(-1, 1, X.shape[1])

        return self.model.predict(X_scaled, verbose=0).flatten()


def get_deep_learning_configs() -> Dict[str, Callable]:
    """
    Get deep learning model factory functions.

    Returns
    -------
    configs : dict
        Dictionary mapping model names to factory functions
    """
    return {
        'MLP': lambda: KerasModelWrapper(
            model_type='MLP',
            hidden_dims=[128, 64, 32],
            epochs=30,
            batch_size=512,
        ),
        'LSTM': lambda: KerasModelWrapper(
            model_type='LSTM',
            hidden_dims=[64, 32],
            epochs=30,
            batch_size=512,
        ),
    }
