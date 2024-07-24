from Tools import Order
import numpy as np
from math import log

class TradingEnvIAR(Order):
    def __init__(self, data, window_size=20):
        self.window_size = window_size
        self.data = data
        self.action_size = 3
        self.max_steps = np.shape(data)[0] - 1
        self.initial_step = self.window_size
        self.state_size = (self.initial_step,np.shape(data[:,1:])[1])
        self.current_step = self.initial_step
        self.position = 0
        self.historic_position = np.zeros((self.initial_step,1))
        self.done = False
        self.orders = []

        self.historic_action = np.full((self.initial_step,1),0)
        self.wallet = 0
        self.reward = 0


    def reset(self):
        self.initial_step = self.window_size
        self.current_step = self.initial_step
        self.done = False

        self.wallet = 0
        self.reward = 0

        self.orders = []
        self.position = 0
        self.historic_position = np.zeros((self.initial_step,1))
        self.historic_action = np.full((self.initial_step,1),0)
        self.historic_wallet = np.full((self.initial_step,1),0)

        state = self.data[:self.initial_step].copy()
        state = np.concatenate((state,self.historic_action,self.historic_wallet),axis = 1)
        self.state_size = np.shape(state[:,1:])

        return state[:,1:]

    def hold(self):
        pass

    def step(self,action):
        if self.done:
            raise ValueError(
                "La partie est déjà terminée. Appelez la méthode reset() pour recommencer."
            )

        if self.position == 0:
          if action == 0:
            self.reward = 0
            self.orders.append(Order())
            self.orders[-1].place_order(self.current_step, 1)
            self.position = self.orders[-1].order_type
          elif action == 1:
            self.reward = 0
            self.orders.append(Order())
            self.orders[-1].place_order(self.current_step, -1)
            self.position = self.orders[-1].order_type
          else :
           self.hold()

        if self.position == 1:
          if action == 0:
            self.hold()
          elif action == 1:
            self.orders[-1].close_order(self.current_step)
            self.orders.append(Order())
            self.reward = 0
            self.orders[-1].place_order(self.current_step, -1)
            self.position = self.orders[-1].order_type
          else :
            self.orders[-1].close_order(self.current_step)
            self.position = 0
            self.reward = 0

        if self.position == -1:
          if action == 1:
            self.hold()
          elif action == 0:
            self.orders[-1].close_order(self.current_step)
            self.orders.append(Order())
            self.reward=0
            self.orders[-1].place_order(self.current_step, 1)
            self.position = self.orders[-1].order_type
          else :
            self.orders[-1].close_order(self.current_step)
            self.position = 0
            self.reward = 0

        self.current_step += 1
        reward = self.calculate_reward(action)
        self.wallet = reward
        #self.historic_position = np.concatenate((self.historic_position, np.array([[self.position]])),axis = 0)
        self.historic_wallet= np.concatenate((self.historic_wallet, np.array([[self.wallet]])),axis = 0)
        self.historic_action = np.concatenate((self.historic_action, np.array([[action]])),axis = 0)
        
        E = self.data[self.current_step-self.initial_step:self.current_step].copy()
        #U = self.historic_position[self.current_step-self.initial_step:].copy()
        I = self.historic_action[self.current_step-self.initial_step:].copy()
        R = self.historic_wallet[self.current_step-self.initial_step:].copy()

        #print(np.shape(E),np.shape(U),np.shape(I))

        next_state = np.concatenate((E,I,R) , axis = 1)

        if self.current_step >= self.max_steps :
            self.done = True
            if len(self.orders) != 0 and self.orders[-1].end_date == 0 :
                self.orders[-1].close_order(self.current_step)
                self.position = 0
                reward = self.calculate_reward(action)
        return next_state[:,1:], reward, self.done, action,{}

    def calculate_reward(self,action):
        if len(self.orders)==0:
            return 0
        else:
           s = 0
           for order in self.orders:
              if order.end_date==0:
                 s += 0
              else:
                 s += order.order_type*(self.data[order.end_date][0]-self.data[order.start_date][0])
           reward=s
           return reward