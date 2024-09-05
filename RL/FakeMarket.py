import random as rd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class News:
    def __init__(self):
        self.value = 0
        self.duration = 0
        self.edit = 0
    
    def generate_random_news(self, current_iteration):
        # Méthode de classe pour générer aléatoirement une nouvelle
        value = round(rd.normalvariate(0, 1.5))
        while value > 3 or value < -3 or value ==0:
            value = round(rd.normalvariate(0, 1.5))
        duration = abs(round(rd.normalvariate(5,2)))
        while duration <0 :
            duration = abs(round(rd.normalvariate(5,5)))
        self.value = value
        self.duration = duration
        self.edit = current_iteration


class StockMarket(News) :
    def __init__(self,nb_news):
        
        self.nb_news = nb_news
        self.iter = 0
        self.price = 1
        self.n_price = 1
        self.news = [News() for _ in range(self.nb_news)]
        self.labels = ['Price']+['Indicator_'+str(i) for i in range(self.nb_news)]
        self.price_history = []
        self.n_price_history = []
        self.news_history = []
        
        ponderation = [rd.uniform(0, 1) for _ in range(self.nb_news)]
        self.ponderation = [p / sum(ponderation) for p in ponderation]
    
    def reset(self):
        self.iter = 0
        self.price = 1
        self.n_price = 1
        self.news = [News() for _ in range(self.nb_news)]
        self.price_history = []
        self.n_price_history = []
        self.news_history = []
        
        ponderation = [rd.uniform(0, 1) for _ in range(self.nb_news)]
        self.ponderation = [p / sum(ponderation) for p in ponderation]
        
    def step(self):
        for news_item in self.news:
            if abs(news_item.edit-self.iter)>news_item.duration:
                news_item.generate_random_news(self.iter)
        for i,news_item in enumerate(self.news):
            noise = rd.normalvariate(0, 0.00065)
            #if abs(noise)>0.001:
                #noise = rd.normalvariate(0, 0.001)
            self.price = self.price + 0.001 *self.ponderation[i] *news_item.value
            event = int(rd.normalvariate(2, 7))
            while event < 2 or event>5:
                event = int(rd.normalvariate(2, 7))

            if self.iter%event==0:
                self.n_price = self.price + noise
            else:
                self.n_price = self.price
        self.news_history.append([news_item.value for news_item in self.news])
        self.price_history.append(self.price)
        self.n_price_history.append(self.n_price)
        self.iter += 1

    def run(self, nb_tour):
        for i in range(nb_tour):
            self.step()
    
    def generate(self,size):
        r = False
        while not r:
            
            self.run(size)
            plt.plot(self.price_history[:4000])
            plt.show()
            #plt.plot(self.price_history[:250])
            plt.plot(self.n_price_history[:250])
            plt.show()
            response = input('Is the genrated data okay for you?(y/n)')
            if response == 'y':
                r = True
            else:
                r = False
                self.reset()
        News = np.array(self.news_history)
        
        Price = np.array(self.price_history)
        Price = np.reshape(Price,(size,1))
        DATASET_FAKE_MARKET = np.concatenate((Price,News),axis=1)
        training = DATASET_FAKE_MARKET[:(size//4)*3]
        test = DATASET_FAKE_MARKET[(size//4)*3:]
        training = pd.DataFrame(training, columns=self.labels)
        test = pd.DataFrame(test, columns=self.labels)
        training.to_csv('C:/Users/Ugo/Documents/AI/Forex_ML/RL/DATA/FAKE_DATA_TRAIN.csv',index=False)
        test.to_csv('C:/Users/Ugo/Documents/AI/Forex_ML/RL/DATA/FAKE_DATA_TEST.csv',index=False)

        Noise_Price = np.array(self.n_price_history)
        Noise_Price = np.reshape(Noise_Price,(size,1))
        N_DATASET_FAKE_MARKET = np.concatenate((Noise_Price,News),axis=1)
        n_training = N_DATASET_FAKE_MARKET[:(size//4)*3]
        n_test = N_DATASET_FAKE_MARKET[(size//4)*3:]
        n_training = pd.DataFrame(n_training, columns=self.labels)
        n_test = pd.DataFrame(n_test, columns=self.labels)
        n_training.to_csv('C:/Users/Ugo/Documents/AI/Forex_ML/RL/DATA/NOISY_FAKE_DATA_TRAIN.csv',index=False)
        n_test.to_csv('C:/Users/Ugo/Documents/AI/Forex_ML/RL/DATA/NOISY_TEST_FAKE_DATA_TEST.csv',index=False)

size = 40000

SM = StockMarket(50)
SM.generate(size)


