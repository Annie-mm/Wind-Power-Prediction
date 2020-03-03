import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class VarRelation:
    """Study the relationship between variables in dataset"""
    
    def __init__(self):
        try:
            self.data = pd.read_pickle('data_by_mean.pkl')
        except:
            raise ImportError("Missing file: 'data_by_mean.pkl")
            
        self.columns = self.data.columns
        
        
    def get_unique_values(self, col=None): 
        """Get unique values of a selected feature"""
        
        
        self.unique = self.data[col].unique()
        return self.unique
    
        
    def plot_moving_average(self, cols='POWER', hours=400):
        """Plot moving average for selected columns and hours"""
        
        if type(cols) == list:
            try:
                for col in cols:
                    self.data['{}-{}MA'.format(col, str(hours))] = self.data[col].rolling(hours).mean()
                    
                self.data.iloc[:,-len(cols):].dropna().plot(
                    title='{} hour moving average'.format(str(hours)),
                    logy=True,
                    grid=True,
                    )
            except:
                raise KeyError('Unvalid column name(s) entered')
                    
                
        elif type(cols) == str:
            try:
                self.data['{}-{}MA'.format(cols, str(hours))] = self.data[cols].rolling(hours).mean()
                
                self.data.iloc[:,-1:].dropna.plot(
                    title='{} hour moving average'.format(str(hours)),
                    logy=True,
                    grid=True,
                    )
            except:
                raise KeyError('Unvalid column name(s) entered')
                                 
        else:
            raise ValueError('Unvalid input-format. Takes list of str or str')
        
    def group_by_time(self, col=None):
        """Group features by time, and visualize"""
        
        self.group = self.data.drop(['PRESS'], axis=1).groupby(pd.Grouper(freq='M')).mean()
        #self.group = self.data[col].groupby(by=[self.data.index.day])
        
        self.group.plot()
        plt.show()
        
        
    def plot_correlation_matrix(self):
        """Plot a correlation matrix"""
        
        self.corr = self.data.corr()
    
        
        fig = plt.figure(figsize=(19, 15))
        ax = fig.add_subplot(111)
        labels = self.data.columns
        
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        
        
        plt.title('Correlation Matrix', fontsize=20)
        
        ax.matshow(self.corr, cmap=plt.cm.RdYlGn,)
        
        
    def plot_numeric_correlation(self):
        self.corr = self.data.corr()
        
        
        sns.heatmap(self.corr, cmap="RdYlGn",
                    xticklabels=self.corr.columns.values,
                    yticklabels=self.corr.columns.values,
                    annot=True, square=True)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        #plt.
        plt.title('Correlation Matrix', fontsize=20)
        plt.show()
        
    
    def plot_numeric_covariance(self):
        self.cov = self.data.cov()
        
        
        sns.heatmap(self.cov, cmap="RdYlGn",
                    xticklabels=self.data.columns.values,
                    yticklabels=self.data.columns.values,
                    annot=True,)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.title('Covariance Matrix', fontsize=20)
        plt.show()
        
    
if __name__ == '__main__':
    c = VarRelation()
    #c.plot_moving_average(cols=['WS10', 'WS100', 'POWER'], hours=400)
    #c.group_by_time()
    #c.plot_correlation_matrix()
    c.plot_numeric_correlation()
    #c.plot_numeric_covariance()