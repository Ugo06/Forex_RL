import numpy as np
import random as rd

from math import log
from utils.tools import Order

class TradingEnv(Order):
    def __init__(self, data: np.array, window_size: int = 20, episode_size: int = 250, n: int = 1, initial_step: None = 'random', mode: dict = None,wallet:int=0, reward_function: str = "default"):
        
        self.data = data

        self.n = n
        self.episode_size = episode_size
        self.window_size = window_size
        self.action_size = 3
        self.reward_function = self._get_reward_function(reward_function=reward_function)

        self.mode = mode if mode is not None else {
            'include_price': False,
            'include_historic_position': False,
            'include_historic_action': False,
            'include_historic_wallet': False
        }
        print(mode['include_historic_position'])

        if type(initial_step) ==str:
            if initial_step == 'random':
                self.initial_step = rd.randint(self.window_size, len(self.data) - (self.n * self.episode_size + 1))
            elif initial_step == 'sequential':
                self.initial_step = self.window_size
        elif type(initial_step) == int:
            if initial_step>= self.window_size and initial_step<=len(self.data) - (self.n * self.episode_size + 1):
                self.initial_step = initial_step
            else:
                raise ValueError(f"The initial step has to be in [{self.window_size},{initial_step<=len(self.data) - (self.n * self.episode_size + 1)}]")
        else:
            raise ValueError("The initial step has to be a string or an integers")
        
        self.current_step = self.initial_step
        self.state_size = self._calculate_state_size()
      
        self.orders = []
        self.position = 0
        self.initial_wallet = wallet
        self.wallet = self.initial_wallet

        self.historic_position = np.zeros((self.window_size, 1))
        self.historic_action = np.full((self.window_size, 1), 0)
        self.historic_wallet = np.full((self.window_size, 1), self.wallet)

        self.done = False

    def reset(self, initial_step: None = 'random'):
        
        if type(initial_step) ==str:
            if initial_step == 'random':
                self.initial_step = rd.randint(self.window_size, len(self.data) - (self.n * self.episode_size + 1))
            elif initial_step == 'sequential':
                if self.initial_step<=len(self.data) - (self.n * self.episode_size + 1):
                    self.initial_step += self.window_size
                else:
                    self.initial_step = self.window_size
        elif type(initial_step) == int:
            if initial_step>= self.window_size and initial_step<=len(self.data) - (self.n * self.episode_size + 1):
                self.initial_step = initial_step
            else:
                raise ValueError(f"The initial step has to be in [{self.window_size},{initial_step<=len(self.data) - (self.n * self.episode_size + 1)}]")
        else:
            raise ValueError("The initial step has to be a string or an integers")
        
        self.current_step = self.initial_step

        self.wallet = self.initial_wallet
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
        reward = self.reward_function()

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
        self.historic_wallet = np.concatenate((self.historic_wallet, np.array([[self.wallet]])), axis=0)
        self.historic_position = np.concatenate((self.historic_position, np.array([[position]])), axis=0)
        self.historic_action = np.concatenate((self.historic_action, np.array([[action]])), axis=0)

    def _get_reward_function(self,reward_function:str):
        if type(reward_function) == str:
            if reward_function == 'initial_wallet':
                return self.default_reward
            elif reward_function == 'portfolio':
                return self.reward_on_PF
            elif reward_function == 'log_portfolio':
                return self.log_reward_on_PF
            elif reward_function == 'norm_open_position':
                return self.norm_open_position_reward
            elif reward_function == 'open_position':
                return self.open_position_reward
            elif reward_function == 'sortino_reward':
                return self.sortino_ratio_reward
            elif reward_function == 'mean_return':
                return self.mean_return_reward
            elif reward_function == 'sharpe_ratio':
                return self.sharpe_ratio_reward
            elif reward_function == 'long_term_0':
                return self.long_term_reward_0
            elif reward_function == 'long_term_1':
                return self.long_term_reward_1
            else:
                raise ValueError(f"reward function:{reward_function} doesn't exist.")
        else:
            return reward_function
    
    def default_reward(self):
        if self.historic_wallet[-1][0] - self.historic_wallet[-2][0] > 0:
            return 1
        elif self.historic_wallet[-1][0] - self.historic_wallet[-2][0]<0:
            return -1
        else:
            return 0
    
    
    def reward_on_PF(self):
        return self.historic_wallet[-1][0] - self.historic_wallet[-2][0]
    
    def log_reward_on_PF(self):
        if self.historic_wallet[-1][0]<=0:
            return 0
        return log(self.historic_wallet[-1][0]/self.historic_wallet[-2][0])
    
    def norm_open_position_reward(self):
        if len(self.orders)==0:
            return 0 
        else:
            if self.orders[-1].end_date==0:
                current_price = self.data[self.current_step][0]
                opening_price = self.data[self.orders[-1].start_date][0]
                position = self.orders[-1].order_type
                return position*(current_price-opening_price)/opening_price
            else:
                return 0
    
    def open_position_reward(self):
        if len(self.orders)==0 or self.orders[-1].end_date != 0:
            return 0 
        else:
            if self.orders[-1].end_date==0:
                current_price = self.data[self.current_step][0]
                opening_price = self.data[self.orders[-1].start_date][0]
                position = self.orders[-1].order_type
                return position*(current_price-opening_price)/0.001
            else:
                return 0
    
    def mean_return_reward(self):
        returns = [self.historic_wallet[i][0] - self.historic_wallet[i-1][0] for i in range(1, len(self.historic_wallet))]
        if not returns:
            return 0
        mean_return = np.mean(returns)
        
        return mean_return
    
    def sharpe_ratio_reward(self):
        returns = [self.historic_wallet[i][0] - self.historic_wallet[i-1][0] for i in range(1, len(self.historic_wallet))]
        if not returns:
            return 0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        return mean_return / std_return if std_return != 0 else 0
    
    def sortino_ratio_reward(self):
        returns = [self.historic_wallet[i][0] - self.historic_wallet[i-1][0] for i in range(1, len(self.historic_wallet))]
        if not returns:
            return 0
        mean_return = np.mean(returns)
        downside_deviation = np.sqrt(np.mean([min(0, r)**2 for r in returns]))
        return mean_return / downside_deviation if downside_deviation != 0 else 0
    
    def long_term_reward_0(self):
        if len(self.orders) == 0 or self.orders[-1].end_date != 0:
            return 0
        
        current_price = self.data[self.current_step][0]
        opening_price = self.data[self.orders[-1].start_date][0]
        position = self.orders[-1].order_type


        current_profit = position * (current_price - opening_price) / 0.001

        max_profit = max([position*(self.data[i][0] - opening_price) / 0.001 for i in range(self.orders[-1].start_date, self.current_step + 1)])

        if current_profit >=max_profit:
            return 1
        elif current_profit >= 0.5 * max_profit and current_profit < max_profit:
            return 0.5
        elif current_profit < 0.5 * max_profit and current_profit >=0:
            return 0
        else:
            return -1
        
    def long_term_reward_1(self):
        if len(self.orders) == 0 or self.orders[-1].end_date != 0:
            self.cumulative_reward = 0
            return 0

        opening_price = self.data[self.orders[-1].start_date][0]
        position = self.orders[-1].order_type

        position_profit = [position*(self.data[i][0] - opening_price) / 0.001 for i in range(self.orders[-1].start_date, self.current_step + 1)]

        # Initialize reward
        if not hasattr(self, 'cumulative_reward'):
            self.cumulative_reward = 0

        # Reward logic
        if position_profit[-1] > position_profit[-2]:
            self.cumulative_reward += 1
        elif position_profit[-1] > position_profit[-2]:
            self.cumulative_reward -= 1

        return self.cumulative_reward
    
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
            price_feature = self.data[self.current_step - self.window_size:self.current_step, 0:1].copy()
            additional_features.append(price_feature)

        if self.mode['include_historic_position']:
            additional_features.append(self.historic_position[-self.window_size:])

        if self.mode['include_historic_action']:
            additional_features.append(self.historic_action[-self.window_size:])

        if self.mode['include_historic_wallet']:
            additional_features.append(self.historic_wallet[-self.window_size:])

        if additional_features:
            additional_features = np.hstack(additional_features)
            return np.hstack((data_state, additional_features))
        else:
            return data_state

