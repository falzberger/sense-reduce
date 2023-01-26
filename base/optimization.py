import os

import keras_tuner as kt
import pandas as pd
import tensorflow as tf

from base.model import Model
from base.training import mse_weighted
from base.window_generator import WindowGenerator
from common import ModelMetadata


class SimpleDense(kt.HyperModel):
    """A simple dense model with few parameters."""

    def __init__(self, window_generator: WindowGenerator, name=None, tunable=True):
        super().__init__(name, tunable)
        self._window = window_generator

    def build(self, hp: kt.HyperParameters) -> tf.keras.Sequential:
        model = tf.keras.Sequential()
        input_length = hp.Int('input_length', min_value=1, max_value=6, step=1)
        model.add(tf.keras.layers.Lambda(lambda x: x[:, -input_length:, :]))
        model.add(tf.keras.layers.Flatten(name='flatten'))

        model.add(tf.keras.layers.Dense(
            units=hp.Int('dense1_units', min_value=8, max_value=32, step=8),
            activation=hp.Choice('dense1_activation', values=['relu', 'tanh', 'linear'], default='relu'),
            name='dense1',
        ))
        model.add(tf.keras.layers.Dense(
            units=self._window.output_shape[0] * self._window.output_shape[1],
            activation='linear',
            name='dense2',
        ))
        model.add(tf.keras.layers.Reshape(
            target_shape=self._window.output_shape,
            name='reshape',
        ))

        learning_rate = hp.Float('learning_rate', min_value=1e-8, max_value=1e-2, sampling='log', default=1e-4)
        choice = hp.Choice('optimizer', ['adam', 'rmsprop'], default='adam')
        if choice == 'adam':
            optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss=mse_weighted,
            metrics=[tf.losses.MeanSquaredError(), tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError()])
        return model


class SimpleLSTM(kt.HyperModel):

    def __init__(self, window_generator: WindowGenerator, recurrent_dropout: bool = True, name=None, tunable=True):
        super().__init__(name, tunable)
        self._window = window_generator
        self.recurrent_dropout = recurrent_dropout

    def build(self, hp: kt.HyperParameters) -> tf.keras.Sequential:
        model = tf.keras.Sequential()
        input_length = hp.Int('input_length', min_value=1, max_value=self._window.input_length, step=6)
        model.add(tf.keras.layers.Lambda(lambda x: x[:, -input_length:, :]))

        if self.recurrent_dropout:
            recurrent_dropout = hp.Float('recurrent_dropout', min_value=0.0, max_value=0.5, step=0.1)
        else:
            recurrent_dropout = 0.0

        model.add(tf.keras.layers.LSTM(
            units=hp.Int('lstm_units', min_value=8, max_value=64, step=8, default=16),
            return_sequences=False,
            recurrent_dropout=recurrent_dropout,
        ))

        dense_dropout = hp.Float('dense_dropout', min_value=0.0, max_value=0.5, step=0.1)
        model.add(tf.keras.layers.Dropout(dense_dropout))
        model.add(tf.keras.layers.Dense(units=self._window.output_shape[0] * self._window.output_shape[1]))
        model.add(tf.keras.layers.Reshape(self._window.output_shape))

        learning_rate = hp.Float('learning_rate', min_value=1e-8, max_value=1e-2, sampling='log', default=1e-4)
        choice = hp.Choice('optimizer', ['adam', 'rmsprop'], default='adam')
        if choice == 'adam':
            optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss=mse_weighted,
            metrics=[tf.losses.MeanSquaredError(), tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError()])
        return model


class ConvLSTM(kt.HyperModel):
    def __init__(self, window_generator: WindowGenerator, name=None, tunable=True):
        super().__init__(name, tunable)
        self._window = window_generator

    def build(self, hp: kt.HyperParameters) -> tf.keras.Sequential:
        inputs = tf.keras.Input(shape=self._window.input_shape)

        conv1_filters = hp.Int('conv1_filters', min_value=8, max_value=32, step=8, default=8)
        conv1 = tf.keras.layers.Conv1D(kernel_size=3, filters=conv1_filters, padding='valid')(inputs)

        conv2_filters = hp.Int('conv2_filters', min_value=16, max_value=32, step=8, default=16)
        conv2 = tf.keras.layers.Conv1D(kernel_size=3, filters=conv2_filters, padding='valid')(conv1)

        maxpool1 = tf.keras.layers.MaxPool1D(pool_size=2)(conv2)

        conv3_filters = hp.Int('conv3_filters', min_value=32, max_value=64, step=16, default=32)
        conv3 = tf.keras.layers.Conv1D(kernel_size=3, filters=conv3_filters, padding='valid')(maxpool1)

        conv4_filters = hp.Int('conv4_filters', min_value=48, max_value=128, step=16, default=64)
        conv4 = tf.keras.layers.Conv1D(kernel_size=3, filters=conv4_filters, padding='valid')(conv3)

        maxpool2 = tf.keras.layers.MaxPool1D(pool_size=2)(conv4)

        lstm_units = hp.Int('lstm_units', min_value=16, max_value=32, step=8, default=24)
        lstm = tf.keras.layers.LSTM(units=lstm_units, return_sequences=False, name='lstm')(maxpool2)

        dense = tf.keras.layers.Dense(units=self._window.output_shape[0] * self._window.output_shape[1],
                                      name='dense')(lstm)
        outputs = tf.keras.layers.Reshape(self._window.output_shape)(dense)

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        learning_rate = hp.Float('learning_rate', min_value=1e-8, max_value=1e-3, sampling='log', default=1e-4)
        choice = hp.Choice('optimizer', ['adam', 'rmsprop'], default='rmsprop')
        if choice == 'adam':
            optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss=mse_weighted,
            metrics=[tf.metrics.MeanSquaredError(), tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError()])

        return model


class ConvStackedLSTM(kt.HyperModel):
    def __init__(self, window_generator: WindowGenerator, recurrent_dropout: bool = True, name=None, tunable=True):
        super().__init__(name, tunable)
        self._window = window_generator
        self.recurrent_dropout = recurrent_dropout

    def build(self, hp: kt.HyperParameters) -> tf.keras.Sequential:
        if self.recurrent_dropout:
            recurrent_dropout = hp.Float('recurrent_dropout', min_value=0.0, max_value=0.5, step=0.25, default=0.0)
        else:
            recurrent_dropout = 0.0

        inputs = tf.keras.Input(shape=self._window.input_shape)
        conv_1_size = hp.Int('conv_1_size', min_value=2, max_value=5, step=1, default=3)
        conv_1 = tf.keras.layers.Conv1D(kernel_size=conv_1_size, filters=16, padding='same')(inputs)

        conv_2_size = hp.Int('conv_2_size', min_value=3, max_value=12, step=1, default=6)
        conv_2 = tf.keras.layers.Conv1D(kernel_size=conv_2_size, filters=16, padding='same')(inputs)

        conv_3_size = hp.Int('conv_3_size', min_value=4, max_value=24, step=3, default=12)
        conv_3 = tf.keras.layers.Conv1D(kernel_size=conv_3_size, filters=16, padding='same')(inputs)

        concat = tf.keras.layers.concatenate([inputs, conv_1, conv_2, conv_3], axis=-1)

        lstm_1 = None
        lstm_1_units = hp.Int('lstm_1_units', min_value=0, max_value=64, step=16, default=0)
        if lstm_1_units > 0:
            lstm_1 = tf.keras.layers.LSTM(lstm_1_units, recurrent_dropout=recurrent_dropout, return_sequences=True
                                          )(concat)

        lstm_2 = None
        lstm_2_units = hp.Int('lstm_2_units', min_value=0, max_value=64, step=16, default=0)
        if lstm_2_units > 0:
            lstm_2 = tf.keras.layers.LSTM(lstm_2_units, recurrent_dropout=recurrent_dropout, return_sequences=True
                                          )(lstm_1 if lstm_1_units > 0 else concat)

        lstm_3_units = hp.Int('lstm_3_units', min_value=16, max_value=128, step=16, default=16)
        lstm_3 = tf.keras.layers.LSTM(lstm_3_units, recurrent_dropout=recurrent_dropout, return_sequences=False
                                      )(lstm_2 if lstm_2_units > 0 else lstm_1 if lstm_1_units > 0 else concat)

        dense_dropout = None
        dense_dropout_rate = hp.Float('dense_dropout', min_value=0.0, max_value=0.5, step=0.25)
        if dense_dropout_rate > 0.0:
            dense_dropout = tf.keras.layers.Dropout(dense_dropout_rate)(lstm_3)

        dense = tf.keras.layers.Dense(units=self._window.output_shape[0] * self._window.output_shape[1],
                                      kernel_initializer=tf.initializers.zeros()
                                      )(dense_dropout if dense_dropout_rate > 0 else lstm_3)
        outputs = tf.keras.layers.Reshape(self._window.output_shape)(dense)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        learning_rate = hp.Float('learning_rate', min_value=1e-8, max_value=1e-3, sampling='log', default=1e-4)
        choice = hp.Choice('optimizer', ['adam', 'rmsprop'], default='rmsprop')
        if choice == 'adam':
            optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss=mse_weighted,
            metrics=[tf.metrics.MeanSquaredError(), tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError()])

        return model


def get_best_epoch(hp: kt.HyperParameters,
                   hm: kt.HyperModel,
                   window: WindowGenerator,
                   ) -> int:
    """Used to find the best number of epochs to train a model without validation data."""
    model: tf.keras.Sequential = hm.build(hp)
    callbacks = [
        # high patience value to prevent early under-fitting
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
    ]
    history = model.fit(
        window.train,
        validation_data=window.val,
        epochs=200,
        callbacks=callbacks)
    val_loss_per_epoch = history.history["val_loss"]
    best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
    print(f"Best epoch: {best_epoch}")
    return best_epoch


def get_best_trained_model(hp: kt.HyperParameters,
                           hm: kt.HyperModel,
                           window: WindowGenerator,
                           ) -> tf.keras.Sequential:
    best_epoch = get_best_epoch(hp, hm, window)
    model: tf.keras.Sequential = hm.build(hp)
    model.fit(
        window.make_dataset(pd.concat([window.train_df, window.val_df], copy=True)),
        epochs=int(best_epoch * 1.2))  # train for a bit longer since we have more data (assuming train_split==0.8)
    return model


def search_model(window: WindowGenerator,
                 hypermodel: kt.HyperModel,
                 tuner: str = 'hyperband',
                 max_epochs=100,
                 patience=5,
                 hyperband_iterations=1,
                 max_trials=100,
                 executions_per_trial=1,
                 model_dir='models',
                 model_name='model',
                 overwrite=False,
                 without_validation=False):
    print(' ⏳ Creating Hyperparameter Search Space...')
    if tuner == 'hyperband':
        tuner = kt.Hyperband(hypermodel,
                             objective='val_loss',
                             max_epochs=max_epochs,
                             factor=3,
                             hyperband_iterations=hyperband_iterations,
                             directory=model_dir,
                             project_name=model_name,
                             overwrite=overwrite,
                             )
    elif tuner == 'bayesian':
        tuner = kt.BayesianOptimization(hypermodel,
                                        objective='val_loss',
                                        max_trials=max_trials,
                                        executions_per_trial=executions_per_trial,
                                        directory=model_dir,
                                        project_name=model_name,
                                        overwrite=overwrite,
                                        )
    else:
        raise ValueError(f'Unknown tuner: {tuner}')
    tuner.search_space_summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                                      mode='min', restore_best_weights=True)
    tuner.search(
        window.train,
        epochs=max_epochs,
        validation_data=window.val,
        callbacks=[early_stopping],
        verbose=2,
    )

    top_n = 3
    print(f' ✅️ Hyperparameter Search Completed! Top {top_n} models: ')
    best_hps = tuner.get_best_hyperparameters(top_n)
    best_models = tuner.get_best_models(top_n)
    for i, hp in enumerate(best_hps):
        print(f'  {i + 1}. {hp.values}')

    print(f'Evaluating best {top_n} models on complete dataset...')
    for i, hp in enumerate(best_hps):
        if without_validation:
            model = get_best_trained_model(hp, hypermodel, window)
        else:
            model = best_models[i]
        val_metrics = model.evaluate(window.val, verbose=2)
        print(f'Model {i + 1} val metrics: {val_metrics}')
        metrics = model.evaluate(window.test, verbose=2)
        print(f'Model {i + 1} test metrics: {metrics}')
        print(model.summary())

        path = os.path.join(model_dir, f'{model_name}_{i + 1}')
        print(f'\nStoring model to {path}...')
        context = {'training': {  # store training metadata
            'stride': window.stride,
            'without_validation': without_validation,
            'val_loss': val_metrics[0],
            'val_mse': val_metrics[1],
            'val_mae': val_metrics[2],
            'val_rmse': val_metrics[3],
            'test_loss': metrics[0],
            'test_mse': metrics[1],
            'test_mae': metrics[2],
            'test_rmse': metrics[3],
        }}
        metadata = ModelMetadata(uuid=f'{model_name}_{i + 1}',
                                 input_features=window.input_features,
                                 input_shape=model.input_shape,
                                 output_features=window.output_features,
                                 output_shape=model.output_shape,
                                 periodicity=window.periodicity,
                                 normalization_mean=window.norm_mean,
                                 normalization_std=window.norm_std,
                                 context=context)
        model = Model(model, metadata)
        model.save_and_convert(path)


if __name__ == '__main__':
    from base.simulator_data import SimulatorData

    sim_data = SimulatorData(
        # 10-minute data for training, using sampling_rate=6
        initial_source='../simulation/zamg/zamg_vienna_hourly.pickle',
        initial_start='2010-01-01',
        initial_end='2019-12-31',
        continual_start='2020-01-01',
        continual_end='2021-12-31',
    )

    EXPERIMENT_NAME = 'zamg_vienna_reduced'
    INPUT_FEATURES = ['TL', 'P', 'RF', 'SO']  # , 'RR_norm', 'FFAM', 'DD_sin', 'DD_cos']
    EXCLUDE_NORMALIZATION = []  # 'RR_norm']
    OUTPUT_FEATURES = ['TL']
    PERIODICITY = ['day', 'year']
    STRIDE = 6
    SAMPLING_RATE = 6

    WINDOW = sim_data.get_window_generator(
        input_features=['TL', 'P', 'RF', 'SO'],
        output_features=['TL'],
        periodicity=['day', 'year'],
        input_length=24 * 5,
        output_length=24,
        stride=6,
        sampling_rate=6,
        batch_size=32,
        validation_split='730d',
    )

    search_model(WINDOW,
                 ConvLSTM(WINDOW),
                 tuner='hyperband',
                 max_epochs=120,
                 patience=30,
                 hyperband_iterations=1,
                 model_dir='../models',
                 model_name=f'zamg_full_201001_201912_ConvLSTM3_search',
                 overwrite=False,
                 )
