import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class VarRelation:
    def __init__(self):
        try:
            self.data = pd.read_pickle('data_by_mean.pkl')
        except:
            print("Missing file: 'data_by_mean.pkl")
    
        self.tp = len(self.data) # Timepoints
        self.columns = self.data.columns
        #self.data['TIME'] = self.data.index
        #self.dtimes = self.data.index.subtract(self.data.index.shift(+1))
        
        self.data['POWER400MA'] = self.data['POWER'].rolling(200).mean()
        self.data['WS10400MA'] = self.data['WS10'].rolling(700).mean()
        self.data['WS100400MA'] = self.data['WS100'].rolling(700).mean()
        
        
    def get_unique_values(self, col=None): 
        #self.unique = list(set(self.data['PRESS'].values.tolist()))
        self.unique = self.data[col].unique()
        return self.unique
        
    def plot_moving_average(self, col='POWER'):
        self.moving_average = self.data[col].rolling(400).mean()
        self.moving_average.dropna().plot()
        
    def group_by_time(self, col=None):
        
        
        self.group = self.data.drop(['PRESS'], axis=1).groupby(pd.Grouper(freq='M')).mean()
        #self.group = self.data[col].groupby(by=[self.data.index.day])
        
        #self.group.plot()
        #plt.show()
        
    def compare_wind_speed(self):
        self.speed = pd.concat([self.data['WS10400MA'], self.data['WS100400MA']], axis=1, keys=['WS10', 'WS100']).dropna()
        self.speed.plot()
        
        
    def plot_correlation_plot1(self):
        self.corr = self.data.corr()
    
        
        f, ax = plt.figure(figsize=(19, 15))
        
        plt.matshow(self.corr, fignum=f.number, cmap='coolwarm', )
        plt.xticks(range(self.data.shape[1]), self.columns, fontsize=14, rotation=45)
        plt.yticks(range(self.data.shape[1]), self.columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix', fontsize=20)
        f.tight_layout()
        
        
    def plot_numeric_correlation(self):
        self.corr = self.data.corr()
        
        
        sns.heatmap(self.corr, 
                    xticklabels=self.corr.columns.values,
                    yticklabels=self.corr.columns.values,
                    annot=True,)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.show()
        
        
    
if __name__ == '__main__':
    c = VarRelation()
    #c.plot_moving_average(col='POWER400MA')
    #c.group_by_time()
    #c.plot_correlation_plot()
    c.plot_numeric_correlation()
    #c.compare_wind_speed()