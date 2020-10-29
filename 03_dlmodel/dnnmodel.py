from numpy.random import seed
from datapreprocess.datasetformodel import ModelPreprocess

import tensorflow as tf

seed(1)

class DnnModel(ModelPreprocess):
    def __init__(self, scaler_x, scaler_y, train_x, train_y, test_x, test_y, test_set_time_dummy):
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.test_set_time_dummy = test_set_time_dummy
        self.model = None


    # DNN
    def model_dnn(self, cols_counts, activation='linear', neurons=32, dropout=0.2,
                  output_activation='linear', kernel_initializer='glorot_normal', n_hidden_layer=2,
                  input_kernel_regularizer=0, dense_kernel_regularizer=0):
        x_len = cols_counts
        input_tensor = tf.keras.Input(shape=(x_len,), name='forecast')
        for i in range(n_hidden_layer):
            # First layer
            if i == 0:
                output_tensor = tf.keras.layers.Dense(neurons, activation=activation,
                                             kernel_initializer=kernel_initializer,
                                             kernel_regularizer=tf.keras.regularizers.l1_l2(input_kernel_regularizer,
                                                                      input_kernel_regularizer))(input_tensor)
                output_tensor = tf.keras.layers.LeakyReLU(alpha=0.1)(output_tensor)
            # 2nd ~ n layer
            else:
                output_tensor = tf.keras.layers.Dense(neurons, activation=activation,
                                             kernel_initializer=kernel_initializer,
                                             kernel_regularizer=tf.keras.regularizers.l1_l2(dense_kernel_regularizer,
                                                                      dense_kernel_regularizer))(output_tensor)
                output_tensor = tf.keras.layers.LeakyReLU(alpha=0.1)(output_tensor)

        output_tensor = tf.keras.layers.Dropout(dropout)(output_tensor)
        output_tensor = tf.keras.layers.Dense(1, activation=output_activation, kernel_initializer=kernel_initializer)(
            output_tensor)
        model = tf.keras.Model(input_tensor, output_tensor)
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Nadam(),
                      metrics=['mean_squared_error', 'mean_absolute_error'])
        model.summary()
        return model

    def model_preformance_and_result(self, batch_size=64, activation='relu', nuerons=32,
                                     dropout=0, output_activation='linear', kernel_initializer='glorot_normal',
                                     n_hidden_layer=3, input_kernel_regularizer=0, dense_kernel_regularizer=0):
        # Callback
        callbacks_list = [
            # keras.callbacks.EarlyStopping(
            #     monitor='mean_squared_error',
            #     patience=10,
            # ),
            # keras.callbacks.ReduceLROnPlateau(
            #     monitor='mean_squared_error',
            #     factor=0.1,
            #     patience=3,
            #     mode='auto',
            #     min_lr=0.01,
            #     verbose=1
            # ),
            # keras.callbacks.TensorBoard(log_dir='./tmp/model_dnn')
        ]

        #Model fitting
        self.model = DnnModel(self.scaler_x, self.scaler_y, self.train_x, self.train_y, self.test_x, self.test_y,
                              self.test_set_time_dummy).model_dnn(self.train_x.shape[1], activation, nuerons, dropout,
                                                                  output_activation, kernel_initializer, n_hidden_layer,
                                                                  input_kernel_regularizer, dense_kernel_regularizer)
        self.model.fit({'forecast': self.train_x}, self.train_y, epochs=100, validation_split=0.2,
                       batch_size=batch_size, verbose=1, callbacks=callbacks_list)
        return self.model