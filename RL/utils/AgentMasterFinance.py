import numpy as np
import tensorflow as tf
import random
from utils.tools import ReplayBuffer

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, Conv2D, MaxPooling2D,Reshape, Input
from tensorflow.keras.optimizers import Adam


class DQNTrader:
    def __init__(self, state_size:int, 
                 action_size:int,type:float='lstm', 
                 config_layer:list=[128,16],
                 batch_size:int=16,buffer_size:int=40000, 
                 gamma:float = 0.995,
                 alpha:float = 1e-5,
                 espilon:float= 1, 
                 epsilon_decay:float = 0.95, 
                 epsilon_min:float= 0.01):
        
        self.state_size = state_size
        self.action_size = action_size

        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = espilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        if type=='lstm':
            self.model = self.build_LSTM_model(lstm_layer=config_layer)
        elif type=='cnn':
            self.model = self.build_CNN_model(conv_layer=config_layer)
        else :
            raise ValueError("Choose the model type between:'lstm','cnn'")
        self.target_model = self.model
        print(self.model.summary())
    
    
    def build_LSTM_model(self,lstm_layer):

        input_state = Input(shape=(self.state_size[0],self.state_size[1]))

        x = LSTM(lstm_layer[0], return_sequences=True, stateful=False)(input_state)
        for i in range(1,len(lstm_layer)):
            if i <len(lstm_layer)-1:
                x = LSTM(lstm_layer[1], return_sequences=True, stateful=False)(x)
            else:
                x = LSTM(lstm_layer[1], return_sequences=False, stateful=False)(x)
        dense_layer1 = Dense(64, activation='linear')(x)
        dropout_layer1 = Dropout(0.5)(dense_layer1)
        dense_layer2 = Dense(4, activation='linear')(dropout_layer1)
        dropout_layer2 = Dropout(0.5)(dense_layer2)
        output_q_values = Dense(self.action_size, activation='linear')(dropout_layer2)

        model = Model(inputs=input_state, outputs=output_q_values)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha), metrics=['accuracy'])
    
        return model
    
    def build_CNN_model(self, conv_layer):

        input_state = Input(shape=(self.state_size[0], self.state_size[1], 1))
        x = Conv2D(filters=conv_layer[0]['filters'], kernel_size=conv_layer[0]['kernel_size'], activation='elu',padding='same')(input_state)
        if 'pool_size' in conv_layer[0]:
                x = MaxPooling2D(pool_size=conv_layer[0]['pool_size'])(x)
        for i in range(1, len(conv_layer)):
            x = Conv2D(filters=conv_layer[i]['filters'], kernel_size=conv_layer[i]['kernel_size'], activation='elu',padding='same')(x)
            if 'pool_size' in conv_layer[i]:
                x = MaxPooling2D(pool_size=conv_layer[i]['pool_size'])(x)
        
        x = Flatten()(x)
        
        dense_layer1 = Dense(32, activation='elu')(x)
        dropout_layer1 = Dropout(0.5)(dense_layer1)
        dense_layer2 = Dense(16, activation='elu')(dropout_layer1)
        dropout_layer2 = Dropout(0.5)(dense_layer2)
        
        output_q_values = Dense(self.action_size, activation='linear')(dropout_layer2)

        model = Model(inputs=input_state, outputs=output_q_values)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha), metrics=['accuracy'])
        
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)

        else :
            act_values = self.model.predict(state,verbose=0)[0]
            return np.argmax(act_values)

    def replay(self):
        minibatch = self.memory.sample(self.batch_size)

        _states = np.array([transition[0][0] for transition in minibatch])
        _actions = np.array([transition[1] for transition in minibatch])
        _rewards = np.array([transition[2] for transition in minibatch])
        _next_states = np.array([transition[3][0] for transition in minibatch])
        _dones = np.array([transition[4] for transition in minibatch])

        targets_1 = self.model.predict(_states, verbose=0)
        next_q_values_1 = self.target_model.predict(_next_states, verbose=0)
        targets_1[range(self.batch_size), _actions] = _rewards + (1 - _dones) * self.gamma * np.amax(next_q_values_1, axis=1)
        
        targets_2 = self.target_model.predict(_states, verbose=0)
        next_q_values_2 = self.model.predict(_next_states, verbose=0)
        targets_2[range(self.batch_size), _actions] = _rewards + (1 - _dones) * self.gamma * np.amax(next_q_values_2, axis=1)
        
        if random.choice([0,1])==0:
            self.model.fit(_states, targets_1, epochs=1, verbose=0)
        else:
            self.model.fit(_states, targets_2,epochs=1,verbose=0)

    def pretrain_supervised(self, supervised_data, epochs=5, batch_size=32):
        for _ in range(epochs):
            _states = np.array([data[0] for data in supervised_data])
            _target_actions = np.array([data[1] for data in supervised_data])
            _rewards = np.array([data[2] for data in supervised_data])
            _next_states = np.array([data[3] for data in supervised_data])
            _dones = np.array([data[4] for data in supervised_data])

            target_q_values = self.model.predict(_states, verbose=0)
            next_q_values = self.model.predict(_next_states, verbose=0)
            max_next_q_values = np.amax(next_q_values, axis=1)

            target_q_values[range(len(_target_actions)), _target_actions] = _rewards + (1 - _dones) * self.gamma * max_next_q_values
            self.model.fit(_states, target_q_values, epochs=1, batch_size=batch_size, verbose=1)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Using CPU.")