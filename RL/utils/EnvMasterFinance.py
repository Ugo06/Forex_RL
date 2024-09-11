import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from math import log
from utils.tools import Order

class TradingEnv(Order):
    def __init__(self, 
                 data: np.array,
                 nb_action:int=3, 
                 window_size: int = 20, 
                 episode_size: int = 250, 
                 n: int = 1, 
                 initial_step: None = 'random',
                 mode: dict = None,
                 wallet:int=0, 
                 reward_function: str = "default",
                 zeta:float=1,
                 beta:float=1):
        
        self.data = data

        self.n = n
        self.episode_size = episode_size
        self.window_size = window_size
        self.action_size = nb_action
        self.reward_function = self._get_reward_function(reward_function=reward_function)

        self.mode = mode if mode is not None else {
            'include_price': False,
            'include_historic_position': False,
            'include_historic_action': False,
            'include_historic_wallet': False,
            'include_historic_orders': False,
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
        self.historic_action = np.full((self.window_size, 1), 2)
        self.historic_wallet = np.full((self.window_size, 1), self.wallet)
        self.historic_orders = np.full((self.window_size,1),0)

        self.done = False

        self.zeta = zeta
        self.beta = beta

    def reset(self, 
              initial_step: None = 'random',
              pas:int=1):
        
        if type(initial_step) ==str:
            if initial_step == 'random':
                self.initial_step = rd.randint(self.window_size, len(self.data) - (self.n * self.episode_size + 1))
            elif initial_step == 'sequential':
                if self.initial_step<=len(self.data) - (self.n * self.episode_size + 1):
                    self.initial_step += pas
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
        self.historic_action = np.full((self.window_size, 1), 2)
        self.historic_wallet = np.full((self.window_size, 1), self.wallet)
        self.historic_orders = np.full((self.window_size,1),0)

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
        self.wallet += self.position * (current_price - previous_price)/0.001
        
        if len(self.orders) == 0:
            value_order = 0
        else:
            if self.orders[-1].end_date == 0:
                value_order = self.orders[-1].order_type*(self.data[self.current_step][0]-self.data[self.orders[-1].start_date][0])
            else:
                value_order = 0
        
        self.historic_wallet = np.concatenate((self.historic_wallet, np.array([[self.wallet]])), axis=0)
        self.historic_position = np.concatenate((self.historic_position, np.array([[position]])), axis=0)
        self.historic_action = np.concatenate((self.historic_action, np.array([[action]])), axis=0)
        self.historic_orders = np.concatenate((self.historic_orders, np.array([[value_order]])), axis=0)

    def _get_reward_function(self,reward_function:str):
        if type(reward_function) == str:
            if reward_function == 'default_return':
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
                return self.mean_reward
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
    
    def mean_reward(self):

        if len(self.orders) == 0:
            duration = 0
        else:
            duration = []
            for order in self.orders:
                if order.end_date != 0:
                    duration.append(order.end_date-order.start_date)
                else:
                    duration.append(self.current_step-order.start_date)
            duration = np.mean(duration)
        
        returns = [self.historic_wallet[i][0] - self.historic_wallet[i-1][0] for i in range(1, len(self.historic_wallet))]
        returns = returns[-5:]
        if not returns:
            return 0
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        reward = self.zeta*mean_return*log(duration+1)
        return reward
    
    def sharpe_ratio_reward(self):
        
        if len(self.orders) == 0:
            duration = 0
        else:
            duration = []
            for order in self.orders:
                if order.end_date != 0:
                    duration.append(order.end_date-order.start_date)
                else:
                    duration.append(self.current_step-order.start_date)
            duration = np.mean(duration)
        
        returns = [self.historic_wallet[i][0] - self.historic_wallet[i-1][0] for i in range(1, len(self.historic_wallet))]
        returns = returns[-5:]
        if not returns:
            return 0
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        reward = self.zeta*mean_return*log(duration+1)
        return reward/std_return if std_return != 0 else 0
    
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

        #max_profit = max([position*(self.data[i][0] - opening_price) / 0.001 for i in range(self.orders[-1].start_date, self.current_step)])
        return current_profit
        
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
        if self.mode['include_historic_orders']:
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
        
        if self.mode['include_historic_orders']:
            additional_features.append(self.historic_orders[-self.window_size:])

        if additional_features:
            additional_features = np.hstack(additional_features)
            return np.hstack((data_state, additional_features))
        else:
            return data_state
    
    def _render_agent_actions(self, save_as):
        fig, ax = plt.subplots()
        data = self.data[self.initial_step-self.window_size:self.current_step, 0]

        # Create a DataFrame to store the agent's actions
        agent_log = pd.DataFrame({
            'time': [t for t in range(len(data))],
            'price': data,
        })

        # Initialize position_open and position_close with NaN values
        position_open = [np.nan for _ in agent_log['time']]
        position_close = [np.nan for _ in agent_log['time']]
        position_type = [np.nan for _ in agent_log['time']]

        # Log the open and close positions based on the orders

        for order in self.orders:
            start_time = order.start_date - self.initial_step + self.window_size - 1
            end_time = order.end_date - self.initial_step + self.window_size - 1
            position_open[end_time] = start_time
            position_close[end_time] = end_time
            position_type[end_time] = order.order_type

        agent_log['position_open'] = position_open
        agent_log['position_close'] = position_close
        agent_log['action'] = position_type

        # Create market data DataFrame
        market_data = pd.DataFrame({
            'time': [t for t in range(len(data))],
            'price': data
        })

        time_steps = market_data['time']
        prices = market_data['price']

        # Initialize lists for the plot
        xdata, ydata = [], []

        # Initialize the market price line
        ln, = ax.plot([], [], 'k-', animated=True, label='Market Price')
        i = 0
        def init():
            ax.set_xlim(min(time_steps), max(time_steps))
            ax.set_ylim(min(prices) * 0.999, max(prices) * 1.001)
            ax.set_title("Agent Trading Actions with Positions Duration")
            ax.set_xlabel("Time")
            ax.set_ylabel("Price (EUR/USD)")
            return ln,

        def update(frame):
            xdata.append(time_steps.iloc[frame])
            ydata.append(prices.iloc[frame])
            ln.set_data(xdata, ydata)

            if frame in agent_log['time'].values:
                current_action = agent_log[agent_log['time'] == frame].iloc[0]
                current_time = current_action['time']
                pos_open = current_action['position_open']
                pos_close = current_action['position_close']

                    
                if pd.notna(pos_open) and pd.notna(pos_close):
                    open_time = pos_open
                    close_time = pos_close
                    open_price = agent_log[agent_log['time'] == open_time]['price'].values[0]
                    close_price = agent_log[agent_log['time'] == close_time]['price'].values[0]
                    open_action = agent_log[agent_log['time'] == close_time]['action'].values[0]

                    # Plot buy (green) and sell (red) points and line for position held
                    if open_action == 1:  # Buy
                        ax.plot(open_time, open_price*0.9995, marker='$↑$', color='green', markersize=15, label="Buy")
                        ax.plot(open_time, open_price, 'go', markersize=5)
                        if open_action*(close_price-open_price)>=0:
                            ax.plot([open_time, close_time], [open_price, close_price], 'g--', label="Long Position",linewidth=3)
                        else:
                            ax.plot([open_time, close_time], [open_price, close_price], 'r--', label="Long Position",linewidth=3)
                        ax.plot(close_time, close_price, 'yo', markersize=5, label="Close")
                    elif open_action == -1:  # Sell
                        ax.plot(open_time, open_price*1.0005, marker='$↓$', color='red', markersize=15, label="Sell")
                        ax.plot(open_time, open_price, 'go', markersize=5)
                        if open_action*(close_price-open_price)>=0:
                            ax.plot([open_time, close_time], [open_price, close_price], 'g--', label="short Position",linewidth=3)
                        else:
                            ax.plot([open_time, close_time], [open_price, close_price], 'r--', label="short Position",linewidth=3)
                        ax.plot(close_time, close_price, 'yo', markersize=5, label="Close")

                    # Plot the closing point as a yellow point
                    

            return ln,

        # Create the animation
        ani = animation.FuncAnimation(
            fig, update, frames=range(len(time_steps)),
            init_func=init, blit=True, repeat=False
        )

        # Save the animation
        ani.save(save_as, writer='ffmpeg')
        print(f"Video saved")

        # Close the figure to avoid memory issues
        plt.close(fig)
