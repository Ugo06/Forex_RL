import numpy as np
import random as rd
from Tools import Order

class TradingEnvIAR(Order):
    def __init__(self, data: np.array, window_size: int = 20, episode_size: int = 250, n: int = 1, initial_step: int = -1):
        self.n = n
        self.episode_size = episode_size
        self.window_size = window_size
        self.data = data
        self.action_size = 3
        
        self.initial_step = rd.randint(self.window_size, len(self.data) - (self.n * self.episode_size + 1)) if initial_step < 0 else initial_step
        
        self.state_size = (self.window_size, np.shape(data[:, 1:])[1])
        self.current_step = self.initial_step
        self.position = 0
        self.historic_position = np.zeros((self.window_size, 1))
        self.done = False
        self.orders = []
        self.historic_action = np.full((self.window_size, 1), 0)
        self.wallet = 0

    def reset(self, initial_step: int = -1):
        self.initial_step = rd.randint(self.window_size, len(self.data) - (self.n * self.episode_size + 1)) if initial_step < 0 else initial_step
        self.current_step = self.initial_step
        self.done = False
        self.wallet = 0
        self.orders = []
        self.position = 0
        self.historic_position = np.zeros((self.window_size, 1))
        self.historic_action = np.full((self.window_size, 1), 0)
        self.historic_wallet = np.full((self.window_size, 1), self.wallet)

        state = self.data[self.initial_step - self.window_size:self.initial_step].copy()
        self.state_size = np.shape(state[:, 1:])
        return state[:, 1:]

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
        self._update_wallet()

        next_state = self.data[self.current_step - self.window_size:self.current_step].copy()
        reward = self.calculate_reward()

        if self.current_step - self.initial_step >= self.episode_size:
            self.done = True
            if self.orders and self.orders[-1].end_date == 0:
                self.orders[-1].close_order(self.current_step)
                self.position = 0

        return next_state[:, 1:], reward, self.done, action, {}

    def _open_position(self, action):
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

    def _handle_long_position(self, action):
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

    def _handle_short_position(self, action):
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

    def _update_wallet(self):
        current_price = self.data[self.current_step][0]
        previous_price = self.data[self.current_step - 1][0]
        self.wallet += self.position * (current_price - previous_price) / 0.001
        self.historic_wallet = np.concatenate((self.historic_wallet, np.array([[self.wallet]])), axis=0)

    def calculate_reward(self):
        return self.historic_wallet[-1][0] - self.historic_wallet[-2][0]
