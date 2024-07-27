import numpy as np
import random as rd
from Tools import Order

class TradingEnv(Order):
    def __init__(self, data: np.array, window_size: int = 20, episode_size: int = 250, n: int = 1, initial_step: int = -1, mode: dict = None):
        
        self.data = data

        self.n = n
        self.episode_size = episode_size
        self.window_size = window_size
        self.action_size = 3

        self.mode = mode if mode is not None else {
            'include_price': False,
            'include_historic_position': False,
            'include_historic_action': False,
            'include_historic_wallet': False
        }

        self.initial_step = rd.randint(self.window_size, len(self.data) - (self.n * self.episode_size + 1)) if initial_step < 0 else initial_step
        self.current_step = self.initial_step
        self.state_size = self._calculate_state_size()
        
        self.historic_position = np.zeros((self.window_size, 1))
        self.historic_action = np.full((self.window_size, 1), 0)
        self.historic_wallet = np.full((self.window_size, 1), self.wallet)

        self.done = False
        self.orders = []
        self.position = 0
        
        self.wallet = 0

    def reset(self, initial_step: int = -1):
        self.initial_step = rd.randint(self.window_size, len(self.data) - (self.n * self.episode_size + 1)) if initial_step < 0 else initial_step
        self.current_step = self.initial_step

        self.wallet = 0
        self.orders = []
        self.position = 0
        
        self.historic_position = np.zeros((self.window_size, 1))
        self.historic_action = np.full((self.window_size, 1), 0)
        self.historic_wallet = np.full((self.window_size, 1), self.wallet)

        state = self._get_state()
        self.state_size = self._calculate_state_size()

        self.done = False
        return state

    def hold(self):
        pass

    def step(self, action):
        if self.done:
            raise ValueError("The episode is already done. Call reset() to start a new episode.")

        if self.position == 0:
            self._open_position(action)
        elif self.position == 1:
            self._handle_long_position(action)
        elif self.position == -1:
            self._handle_short_position(action)

        self.current_step += 1
        self._update_historics(action=action,position=self.position)

        next_state = self._get_state()
        reward = self.calculate_reward()

        if self.current_step - self.initial_step >= self.episode_size:
            self.done = True
            if self.orders and self.orders[-1].end_date == 0:
                self.orders[-1].close_order(self.current_step)
                self.position = 0

        return next_state, reward, self.done, action, {}

    def _open_position(self, action:int):
        if action == 0:
            self.orders.append(Order())
            self.orders[-1].place_order(self.current_step, 1)
            self.position = 1
        elif action == 1:
            self.orders.append(Order())
            self.orders[-1].place_order(self.current_step, -1)
            self.position = -1
        else:
            self.hold()

    def _handle_long_position(self, action:int):
        if action == 0:
            self.hold()
        elif action == 1:
            self.orders[-1].close_order(self.current_step)
            self.orders.append(Order())
            self.orders[-1].place_order(self.current_step, -1)
            self.position = -1
        else:
            self.orders[-1].close_order(self.current_step)
            self.position = 0

    def _handle_short_position(self, action:int):
        if action == 1:
            self.hold()
        elif action == 0:
            self.orders[-1].close_order(self.current_step)
            self.orders.append(Order())
            self.orders[-1].place_order(self.current_step, 1)
            self.position = 1
        else:
            self.orders[-1].close_order(self.current_step)
            self.position = 0

    def _update_historics(self,action:int,position:int):
        current_price = self.data[self.current_step][0]
        previous_price = self.data[self.current_step - 1][0]
        self.wallet += self.position * (current_price - previous_price) / 0.001
        self.historic_position = np.concatenate((self.historic_position, np.array([[position]])), axis=0)
        self.historic_action = np.concatenate((self.historic_action, np.array([[action]])), axis=0)

    def calculate_reward(self):
        return self.historic_wallet[-1][0] - self.historic_wallet[-2][0]
    
    def _calculate_state_size(self):
        state_columns = 0
        if self.mode['include_price']:
            state_columns += 1
        if self.mode['include_historic_position']:
            state_columns += 1
        if self.mode['include_historic_action']:
            state_columns += 1
        if self.mode['include_historic_wallet']:
            state_columns += 1
        return (self.window_size, state_columns + np.shape(self.data[:, 1:])[1])

    def _get_state(self):
        data_state = self.data[self.current_step - self.window_size:self.current_step, 1:].copy()
        additional_features = []

        if self.mode['include_price']:
            price_feature = self.data[self.current_step - self.window_size:self.current_step, 0].copy()
            additional_features.append(price_feature)

        if self.mode['include_historic_position']:
            additional_features.append(self.historic_position[-self.window_size:])

        if self.mode['include_historic_action']:
            additional_features.append(self.historic_action[-self.window_size:])

        if self.mode['include_historic_wallet']:
            additional_features.append(self.historic_wallet[-self.window_size:])

        if additional_features:
            additional_features = np.concatenate(additional_features, axis=1)
            return np.concatenate((data_state, additional_features), axis=1)
        else:
            return data_state

