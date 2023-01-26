import os
from typing import Callable, Optional, List

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from base.model import Model
from base.window_generator import WindowGenerator
from common import ModelMetadata


class ModelDef:

    def __init__(self,
                 name: str,
                 model_fn: Callable[[WindowGenerator], tf.keras.Sequential],
                 optimizer: str,
                 learning_rate: float,
                 fine_tune_rate: float = 1e-6,
                 freeze_layers: List[str] = None,
                 ):
        self.name = name
        self.model_fn = model_fn
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.fine_tune_rate = fine_tune_rate
        self.freeze_layers = freeze_layers
        if freeze_layers is None:
            self.freeze_layers = []


@tf.keras.utils.register_keras_serializable()
def mse_weighted(y_true, y_pred):
    """Custom version of Mean Squared Error (MSE) loss function that penalizes short-term errors more."""
    shape = tf.shape(y_pred)
    penalty = tf.cast(tf.linspace(start=10., stop=tf.cast(1 / shape[1], tf.float32), num=shape[1]), dtype=tf.float32)
    penalty = tf.expand_dims(penalty, axis=-1)
    penalty = tf.expand_dims(tf.transpose(tf.repeat(penalty, repeats=shape[0], axis=1)), axis=-1)
    squared_difference = tf.square(y_true - y_pred) * penalty
    return tf.reduce_mean(squared_difference, axis=-1)


def zamg_dense(window: WindowGenerator) -> tf.keras.Sequential:
    """
    The optimal model found with Hyperband search for SimpleDense (2010-2019 training, 2020-2021 test).

    Parameters: 936

    Metrics (mse_weighted, MSE, MAE, RMSE):
    Validation: [0.23773212730884552, 0.06348775327205658, 0.18336468935012817, 0.2521352171897888]
    Test: [0.26755574345588684, 0.07195460051298141, 0.19036033749580383, 0.26825156807899475]
    """
    input_length = 4
    model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -input_length:, :]),
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(16, activation='relu', name='dense1'),
        tf.keras.layers.Dense(window.output_shape[0] * window.output_shape[1], activation='linear', name='dense2'),
        tf.keras.layers.Reshape(window.output_shape, name='reshape')
    ]
    )
    model.compile(
        optimizer=tf.optimizers.RMSprop(learning_rate=0.002399431372613329),
        loss=mse_weighted,
        metrics=[tf.metrics.MeanSquaredError(), tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError()]
    )

    return model


def zamg_lstm(window: WindowGenerator) -> tf.keras.Sequential:
    """
    The optimal model found with Hyperband search for SimpleLSTM
    without recurrent dropout (2010-2019 training, 2020-2021 test).

    Parameters: ~6040

    Metrics (mse_weighted, MSE, MAE, RMSE):
    Validation: [0.21758656203746796, 0.05991469323635101, 0.1760409027338028, 0.2447098344564438]
    Test: [0.24341025948524475, 0.06680379062891006, 0.18084481358528137, 0.2584553062915802]
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=window.input_shape),
        tf.keras.layers.Lambda(lambda x: x[:, -48:, :]),
        tf.keras.layers.LSTM(32, input_shape=(48, window.input_shape[1]), return_sequences=False, name='lstm'),
        tf.keras.layers.Dropout(0.1, name='dropout'),
        tf.keras.layers.Dense(window.output_length, name='dense'),
        tf.keras.layers.Reshape(window.output_shape, name='reshape'),
    ]
    )
    model.compile(
        optimizer=tf.optimizers.RMSprop(learning_rate=0.0005028709561526049),
        loss=mse_weighted,
        metrics=[tf.metrics.MeanSquaredError(), tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError()]
    )
    return model


def zamg_conv_lstm(window: WindowGenerator) -> tf.keras.Sequential:
    """The best model found with Hyperband search for ConvLSTM (2010-2019 training, 2020-2021 test).
    Considering the rules of thumb for CNN design: channels increase, width and height decrease.
    Similar to the model designed by Kreuzer et al. (2019).

    Parameters: 26,016

    Metrics (mse_weighted, MSE, MAE, RMSE):
    Validation: [0.218229427933693, 0.059479683637619, 0.176438674330711, 0.24386265873909]
    Test: [0.235091030597687, 0.0640134513378143, 0.178532868623734, 0.253029823303223]
    """
    model = tf.keras.Sequential([
        tf.keras.Input(shape=window.input_shape),
        tf.keras.layers.Conv1D(kernel_size=3, filters=24, padding='same', name='conv1'),
        tf.keras.layers.Conv1D(kernel_size=3, filters=32, padding='same', name='conv2'),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Conv1D(kernel_size=3, filters=48, padding='same', name='conv3'),
        tf.keras.layers.Conv1D(kernel_size=3, filters=64, padding='same', name='conv4'),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.LSTM(24, return_sequences=False, name='lstm'),
        tf.keras.layers.Dense(window.output_shape[0] * window.output_shape[1], name='dense'),
        tf.keras.layers.Reshape(window.output_shape, name='reshape')
    ]
    )
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.00024407154977751656),
        loss=mse_weighted,
        metrics=[tf.metrics.MeanSquaredError(), tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError()]
    )
    return model


def train_model(window: WindowGenerator,
                model_func: Callable[[WindowGenerator], tf.keras.Sequential],
                max_epochs=100,
                patience=10,
                without_validation=False,
                model_dir='models',
                model_name='model',
                plot=False,
                ) -> None:
    """Trains a Tensorflow Model and stores it in SavedModel and TFLite format.

    Args:
        window:
        model_func:
        max_epochs:
        patience: Is also used to find out the best number of epochs to train a model without validation data.
        without_validation:
        model_dir:
        model_name:
        plot: Whether to display any plots of the training process or only store them to the model directory.
    """
    path = os.path.join(model_dir, model_name)
    os.makedirs(path, exist_ok=True)
    print(' ⏳ Starting Model Training...')

    model = model_func(window)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min',
                                                      restore_best_weights=True,
                                                      )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     patience=int(patience / 2),
                                                     mode='min',
                                                     factor=0.2,
                                                     verbose=1,
                                                     )
    history = model.fit(window.train,
                        epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping, reduce_lr],
                        )
    for metric in model.metrics_names:
        plot_history(history, metric, path=os.path.join(path, f'history_{metric}.png'), show=plot)

    val_performance = model.evaluate(window.val)
    if without_validation:
        val_loss_per_epoch = history.history["val_loss"]
        best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
        print(f' ⏳ Starting Model Training without Validation Data ({best_epoch} epochs)...')
        model = model_func(window)
        model.fit(
            window.make_dataset(pd.concat([window.train_df, window.val_df], copy=True)),
            epochs=int(best_epoch * 1.2)
        )  # train for a bit longer since we have more data (assuming train_split==0.8)

    test_performance = model.evaluate(window.test)

    if plot:
        for col in window.output_features:
            window.plot(plot_col=col, model=model)

    print(' 📉️ Model Performance:')
    context = {'stride': window.stride, 'sampling_rate': window.sampling_rate}
    print('Validation')
    for name, value in zip(model.metrics_names, val_performance):
        context[f'val_{name}'] = value
        print(f'   - {name}: {value:.4f}')
    print('Test')
    for name, value in zip(model.metrics_names, test_performance):
        context[f'test_{name}'] = value
        print(f'   - {name}: {value:.4f}')
    print(model.summary())

    print(f'\nStoring model to {path}...')
    metadata = ModelMetadata(uuid=model_name,
                             input_features=window.input_features,
                             input_shape=model.input_shape,
                             output_features=window.output_features,
                             output_shape=model.output_shape,
                             periodicity=window.periodicity,
                             normalization_mean=window.norm_mean,
                             normalization_std=window.norm_std,
                             context={'training': context}
                             )
    model = Model(model, metadata)
    model.save_and_convert(path)
    print(f' ✅ Model training finished')


def plot_history(history, metric: str, path: Optional[str] = None, show: bool = True) -> None:
    loss = history.history[metric]
    val_loss = history.history[f'val_{metric}']
    epochs = range(1, len(loss) + 1)
    fig: plt.Figure = plt.figure()
    plt.plot(epochs, loss, 'y', label="Training")
    plt.plot(epochs, val_loss, 'b', label="Validation")
    plt.title(metric)
    plt.legend()
    if show:
        plt.show()
    if path is not None:
        fig.savefig(path)
    plt.close(fig)
