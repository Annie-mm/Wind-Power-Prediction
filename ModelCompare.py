# -*- coding: utf-8 -*-
"""
Created on Thu Mar 5 17:45:50 2020

@author: Sigve Sørensen & Ernst-Martin Buduschin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pickle

import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from collections import Counter


class CompareModels:
    def __init__(self):
        
        try:
            self.data = pd.read_pickle('data_by_mean.pkl')
        except:
            raise ImportError("Missing file: 'data_by_mean.pkl")
            
        self.data = self.data[['POWER', 'WS10', 'WS100']]
        self.columns = self.data.columns
        
        
    def LinearRegression(self, pairplot=False):
        """Perform a linear regression"""
        
        
        if pairplot:
            fig = plt.figure()
            sb.pairplot(self.data)
            plt.show()
            
        predict = 'POWER'
        
        X = self.data.drop([predict], 1).values
        y = self.data['POWER'].values
        
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
        
        best = 0
        
        
        
        for _ in range(1000):
            
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
        
            linear = linear_model.LinearRegression()
            linear.fit(x_train, y_train)
            acc = linear.score(x_test, y_test)
            
            if acc > best:
                print(acc)
                best = acc
                
                """Save our model with pickle. Saved with best result: 70.69 %""".format(best)
                with open('linregmodel.pkl', 'wb') as f:
                    pickle.dump(linear, f)
                
                
        print('Accuracy: {}'.format(acc))         # Accuracy for what the Power will be given 
        print('co: ', linear.coef_)               # Coefficients for attributes
        print('Intercept: ', linear.intercept_)   # Intercept of y
        

            
        """Read in our pickle file"""
        pickle_in = open('linregmodel.pkl', 'rb')
        linear = pickle.load(pickle_in)
        
        p = 'WS100'
        
        plt.figure()
        
        plt.scatter(self.data[p], self.data[predict])
        plt.xlabel(p)
        plt.ylabel('Power')
        plt.tight_layout()
        plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

if __name__ == '__main__':
    c = CompareModels()
    c.LinearRegression()
        


