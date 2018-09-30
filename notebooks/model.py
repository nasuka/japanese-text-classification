from keras.models import Model
from keras.layers import (
    Dense, Embedding, Input, SpatialDropout1D, Reshape, Conv2D, MaxPool2D, Concatenate, Flatten,
    LSTM, Bidirectional, GlobalMaxPool1D, Dropout
)


def get_lstm(
        max_len,
        max_features,
        embed_size=128,
        lstm_unit_size=50,
        hidden_unit_size=50,
        class_num=9
):
    inp = Input(shape=(max_len, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(lstm_unit_size, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(hidden_unit_size, activation="relu")(x)
    x = Dense(class_num, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def get_cnn(
        max_len,
        max_features,
        embed_size=128,
        filter_sizes=[1, 2, 3, 5],
        num_filters=32,
        dropout_rates=[0.45, 0.1],
        class_num=9
):
    inp = Input(shape=(max_len,))
    x = Embedding(max_features, embed_size)(inp)
    x = SpatialDropout1D(dropout_rates[0])(x)
    x = Reshape((max_len, embed_size, 1))(x)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size), kernel_initializer='normal',
                    activation='elu')(x)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size), kernel_initializer='normal',
                    activation='elu')(x)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), kernel_initializer='normal',
                    activation='elu')(x)
    conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size), kernel_initializer='normal',
                    activation='elu')(x)

    maxpool_0 = MaxPool2D(pool_size=(max_len - filter_sizes[0] + 1, 1))(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(max_len - filter_sizes[1] + 1, 1))(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(max_len - filter_sizes[2] + 1, 1))(conv_2)
    maxpool_3 = MaxPool2D(pool_size=(max_len - filter_sizes[3] + 1, 1))(conv_3)

    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
    z = Flatten()(z)
    z = Dropout(dropout_rates[1])(z)

    outputs = Dense(class_num, activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=outputs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def get_mlp(
        input_size,
        hidden_unit_size=100,
        class_num=9
):
    inp = Input(shape=(input_size,))
    hidden = Dense(hidden_unit_size, activation='relu')(inp)
    outputs = Dense(class_num, activation='softmax')(hidden)
    model = Model(inputs=inp, outputs=outputs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model