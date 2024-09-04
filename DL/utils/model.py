import tensorflow as tf


from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, Conv2D, MaxPooling2D,Reshape, Input
from tensorflow.keras.optimizers import Adam


def lstm_model(lstm_layer,alpha,size_feature):
    input_state = Input(shape=(size_feature[0],size_feature[1]))

    x = LSTM(lstm_layer[0], return_sequences=True, stateful=False)(input_state)
    for i in range(1,len(lstm_layer)):
        if i <len(lstm_layer)-1:
            x = LSTM(lstm_layer[1], return_sequences=True, stateful=False)(x)
        else:
            x = LSTM(lstm_layer[1], return_sequences=False, stateful=False)(x)
    dense_layer1 = Dense(16, activation='linear')(x)
    dropout_layer1 = Dropout(0.5)(dense_layer1)
    dense_layer2 = Dense(4, activation='linear')(dropout_layer1)
    dropout_layer2 = Dropout(0.5)(dense_layer2)
    dense_layer_3 = Dense(1, activation='linear')(dropout_layer2)

    model = Model(inputs=input_state, outputs=dense_layer_3)
    model.compile(loss='mse', optimizer=Adam(learning_rate=alpha), metrics=['accuracy'])

    return model


