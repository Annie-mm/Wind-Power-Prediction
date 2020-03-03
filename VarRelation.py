import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

class VarRelation:
    def __init__(self):
        try:
            self.data = pd.read_pickle('data_by_mean.pkl')
        except:
            print("Missing file: 'data_by_mean.pkl")
    
        self.tp = len(self.data) # Timepoints
        #self.data['TIME'] = self.data.index
        #self.dtimes = self.data.index.subtract(self.data.index.shift(+1))
        
        self.data['POWER400MA'] = self.data['POWER'].rolling(400).mean()
        self.data['WS10400MA'] = self.data['WS10'].rolling(500).mean()
        self.data['WS100400MA'] = self.data['WS100'].rolling(500).mean()
        
        
    def get_unique_values(self, col=None): 
        #self.unique = list(set(self.data['PRESS'].values.tolist()))
        self.unique = self.data[col].unique()
        return self.unique
        
    def plot_moving_average(self, col='POWER'):
        self.moving_average = self.data[col].rolling(400).mean()
        self.moving_average.dropna().plot()
        
    def group_by_time(self, col='POWER'):
        
        self.group = self.data[col].groupby(by=[self.data.index.day])
        self.group.plot()
        plt.show()
        
    def compare_wind_speed(self):
        self.speed = pd.concat([self.data['WS10400MA'], self.data['WS100400MA']], axis=1, keys=['WS10', 'WS100']).dropna()
        self.speed.plot()
        
        
        
        
        
    
if __name__ == '__main__':
    c = VarRelation()
    #c.plot_moving_average(column='TEMP2')
    #c.group_by_time(col='POWER400MA')
    c.compare_wind_speed()