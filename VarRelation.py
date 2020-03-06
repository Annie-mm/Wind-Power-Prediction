# -*- coding: utf-8 -*-
"""
Created on Mon Mar 2 13:24:31 2020

@author: Sigve Sørensen & Ernst-Martin Buduschin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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


    def same_day(self, date_string):
        """remove year"""
        return datetime.strptime(date_string, "%Y-%m-%d %h:%m:%s").strftime('%m-%d')
    
        
    def plot_moving_average(self, cols='POWER', hours=400):
        """Plot moving average for selected columns and hours"""
        
        self.ma = self.data.copy()
        
        if type(cols) == list:
            try:
                for col in cols:
                    self.ma['{}-{}MA'.format(col, str(hours))] = self.ma[col].rolling(hours).mean()
                
                self.ma.iloc[:,-len(cols):].dropna().plot(
                    title='{} hour moving average'.format(str(hours)),
                    logy=True,
                    grid=True,
                    )
            except:
                raise KeyError('Unvalid column name(s) entered')
                    
                
        elif type(cols) == str:
            try:
                self.ma['{}-{}MA'.format(cols, str(hours))] = self.ma[cols].rolling(hours).mean()
                
                plt.figure()
                self.ma.iloc[:,-1:].dropna.plot(
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
        
        #.drop(['PRESS'], axis=1)
        
        self.group = self.data.groupby(pd.Grouper(freq='H')).mean()
        #self.group = self.data[col].groupby(by=[self.data.index.day])
        
        self.group.plot()
        plt.show()
        
        
    def plot_correlation_matrix(self):
        """Plot a correlation matrix"""
        
        self.corr = self.data.corr().copy()
    
        
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
        self.corr = self.data.corr().copy()
        
        plt.figure(figsize=(10,8))
        
        sns.heatmap(self.corr, cmap="RdYlGn",
                    xticklabels=self.corr.columns.values,
                    yticklabels=self.corr.columns.values,
                    annot=True, square=True)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.title('Correlation Matrix', fontsize=20)
        plt.show()
        
    
    def plot_numeric_covariance(self):
        self.cov = self.data.cov().copy()
        
        
        sns.heatmap(self.cov, cmap="RdYlGn",
                    xticklabels=self.data.columns.values,
                    yticklabels=self.data.columns.values,
                    annot=True,)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.title('Covariance Matrix', fontsize=20)
        plt.show()
        
    def plot_monthly_power_speed_relation(self, setting='monthly'):
        fig = plt.figure(figsize=(12, 8))
        date = pd.to_datetime(self.data.index)
        data = self.data.filter(['POWER', 'WS100', 'WS10'])
        data['year'] = date.year
        data['month'] = date.month
        months = set(date.month)
        years = set(date.year)
        mean_data = pd.DataFrame()
        for yr in years:
            for mnth in months:
                tmp_data = data[(data['year'] == yr) & (data['month'] == mnth)]
                if not tmp_data.empty:
                    tmp = tmp_data.filter(['POWER', 'WS10', 'WS100']).mean()
                    tmp['TIMESTAMP'] = pd.Timestamp(yr, mnth, 15).month_name()
                    mean_data = mean_data.append(tmp, ignore_index=True)
        mean_data.set_index('TIMESTAMP', inplace=True)
        ax = mean_data.plot(
            ax=fig.gca(),
            kind='bar',
            y=['POWER'],
            # y=['POWER', 'WS100', 'WS10', ],
            # secondary_y=['WS100', 'WS10',],
            position=1,
            width=.4,
            # rot=0
            )
        mean_data.plot(
            ax=ax,
            kind='bar',
            y=['WS10', 'WS100'],
            color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1:3],
            position=0,
            width=.4,
            secondary_y=True, 
            stacked=True,
            rot=45,
            )
        ax.set_ylabel('average normalized power')
        ax.right_ax.set_ylabel('average wind speed (m/s)')
        ax.set_xlabel('month')
        ax.set_xlim([-.75, None])

        
        
if __name__ == '__main__':
    c = VarRelation()
    # c.plot_moving_average(cols=['WS10', 'WS100', 'POWER'], hours=24*30)
    # c.group_by_time()
    # c.plot_correlation_matrix()
    # c.plot_numeric_correlation()
    # c.plot_numeric_covariance()
    c.plot_monthly_power_speed_relation()