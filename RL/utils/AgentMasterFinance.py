import numpy as np
import tensorflow as tf
import random
from utils.tools import ReplayBuffer

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, Conv2D, MaxPooling2D,Reshape, Input
from tensorflow.keras.optimizers import Adam


class DQNTrader:
    def __init__(self, state_size, action_size, lstm_layer = [128,16],buffer_size=40000, gamma = 0.995, espilon= 1, epsilon_decay = 0.95, epsilon_min = 0.01):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = ReplayBuffer(buffer_size)

        
        self.gamma = gamma
        self.epsilon = espilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.model = self.build_LSTM_model(lstm_layer)
        self.target_model = self.model
        print(self.model.summary())
    
    def build_LSTM_model(self,lstm_layer):
        model = Sequential()
        input_state = Input(shape=(self.state_size[0],self.state_size[1]))

        lstm_layer_1 = LSTM(lstm_layer[0], return_sequences=True, stateful=False)(input_state)
        lstm_layer_2 = LSTM(lstm_layer[1], return_sequences=False, stateful=False)(lstm_layer_1)
        dense_layer1 = Dense(32, activation='elu')(lstm_layer_2)
        dropout_layer1 = Dropout(0.5)(dense_layer1)
        dense_layer2 = Dense(16, activation='elu')(dropout_layer1)
        dropout_layer2 = Dropout(0.5)(dense_layer2)
        output_q_values = Dense(self.action_size, activation='linear')(dropout_layer2)

        model = Model(inputs=input_state, outputs=output_q_values)
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)

        else :
            act_values = self.model.predict(state,verbose=0)[0]
            return np.argmax(act_values)

    def replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)

        _states = np.array([transition[0][0] for transition in minibatch])
        _actions = np.array([transition[1] for transition in minibatch])
        _rewards = np.array([transition[2] for transition in minibatch])
        _next_states = np.array([transition[3][0] for transition in minibatch])
        _dones = np.array([transition[4] for transition in minibatch])


        targets_1 = self.model.predict(_states, verbose=0)
        next_q_values_1 = self.target_model.predict(_next_states, verbose=0)
        targets_1[range(batch_size), _actions] = _rewards + (1 - _dones) * self.gamma * np.amax(next_q_values_1, axis=1)
        
        targets_2 = self.target_model.predict(_states, verbose=0)
        next_q_values_2 = self.model.predict(_next_states, verbose=0)
        targets_2[range(batch_size), _actions] = _rewards + (1 - _dones) * self.gamma * np.amax(next_q_values_2, axis=1)
        
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