import random
import pandas as pd
import numpy as np

class Order:
    def __init__(self):
        self.end_date = 0
        self.start_date = 0
        self.order_type = None
        self.gain = 0

    def reset(self):
        self.end_date = 0
        self.start_date = 0
        self.order_type = None
        self.gain = 0

    def place_order(self,date,position):
        self.start_date = date
        self.order_type = position

    def close_order(self,date):
        self.end_date = date

    def historic_order(self,ho):
        L = []
        for i in range(len(ho)):
            A = (ho[i].start_date,ho[i].end_date,ho[i].order_type,ho[i].gain)
            L.append(A)

        return L

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, experience):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
class PrepareData:
    def __init__(self,data):
        self.data = data
    
    def convert_to_npy(self):
        self.data = self.data.to_numpy()
    
    def Sample_1(self,data,size=40):
        L= []
        for i in range(len(data)-size):
            if i%size ==0:
                L.append(i)

        DATA = []
        for i in range(len(L)-1):
            DATA.append(data[L[i]:L[i+1]])
        random.shuffle(DATA)
        return DATA
    
    def Sample_2(self,data,window_size=260,jump=20):
        L= []
        for i in range(len(data)-jump):
            if i%jump ==0 and len(data)-i>window_size:
                L.append(i)

        DATA = []
        for i in range(len(L)):
            DATA.append(data[L[i]:L[i]+window_size])
        random.shuffle(DATA)
        return DATA

    
    def normalize(self):
        self.norm_data = self.data.copy()
        for i in range(np.shape(self.data)[1]):
            maximum = np.max(self.data[:,i])
            minimum = np.min(self.data[:,i])
            if maximum-minimum == 0:
                np.delete(self.norm_data, i, axis=1)
            else:
                self.norm_data[:,i]=(self.data[:,i]-minimum)/(maximum-minimum)
    
    def data_for_training(self,size):
        self.convert_to_npy()
        self.normalize()
        return self.Sample_1(self.norm_data,size=size)
    
    def data_for_test(self):
        self.convert_to_npy()
        self.normalize()
        return self.norm_data
