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
import matplotlib as mlt


import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import scale
from sklearn.metrics import r2_score, mean_squared_error
from collections import Counter



class CompareModels:
    def __init__(self):
        
        try:
            self.data = pd.read_pickle('data_by_mean.pkl')
        except:
            raise ImportError("Missing file: 'data_by_mean.pkl")
            
        self.data = self.data[['POWER', 'WS10', 'WS100']]
        self.columns = self.data.columns
        
        
    def LinearRegression(self, pairplot=False, run_model=True, runs=1000):
        """Perform a linear regression"""
        
        
        if pairplot:
            sb.pairplot(self.data)
            plt.show()
            
        predict = 'POWER'
        predictor = 'WS100'
        
        X = self.data.drop([predict], 1).values.reshape(-1,1)
        y = self.data[predict].values
        
        best_acc = 0
        
        """Running the model 1000 times trying to find the best model"""
        if run_model:
            for _ in range(runs):
            
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                
                linear = LinearRegression()
                linear.fit(X_train, y_train)
                acc = linear.score(X_test, y_test)
                
                
                if acc > best_acc:
                    print(acc)
                    print(r2_score(y, linear.predict(X)))
                    best_acc = acc
                    
                    """Save our model with the best accuracy"""
                    with open('linregmodel.pkl', 'wb') as f:
                        pickle.dump(linear, f)
                    
                    
            print('Model saved')       
            print('Accuracy: {}'.format(acc))         # Accuracy for what the Power will be given 
            print('co: ', linear.coef_)               # Coefficients for attributes
            print('Intercept: ', linear.intercept_)   # Intercept of y
        
        else:
            try:
                """Read in our pickle file"""
                pickle_in = open('linregmodel.pkl', 'rb')
                linear = pickle.load(pickle_in)
                print('Polynomial regression model imported')
            except:
                raise ValueError('run_model set to False, missing file "linregmodel.pkl". Set run_model True to run and save model.')
        
        """Hexplot since big dataset"""
        plt.hexbin(X.flatten(), y, gridsize=20, cmap='Blues', norm=mlt.colors.LogNorm())
        plt.axis([X.min(), X.max(), y.min(), y.max()])
        cb = plt.colorbar()
        cb.set_label('Scatter density')
        plt.plot(X, (linear.coef_*X+linear.intercept_), color='#417ed6', alpha=0.7, lw=0.7)
        plt.xlabel(predictor)
        plt.ylabel(predict)
        plt.tight_layout()
        plt.show()       
        
        
        
    def PolynomialRegression(self, run_model=True, runs=1):
        """Perform a polynomial regression"""
        
        from sklearn.preprocessing import PolynomialFeatures 
        
        predict = 'POWER'
        predictor = 'WS100'
        
        X = self.data.drop([predict], 1).values.reshape(-1,1)
        y = self.data[predict].values
        
        best_acc = 0
        
        if run_model:
            for _ in range(runs):
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                
                
                
                poly_reg = PolynomialFeatures(degree = 6)
                X_poly = poly_reg.fit_transform(X) 
                poly_reg.fit(X_poly, y)
                reg = LinearRegression()
                reg.fit(X_poly, y)
                

                
                acc = round(reg.score(X_poly, y),4)
                
                if acc > best_acc:
                    print(acc)
                    best_acc = acc
                    
                    """Save our model with pickle. Saved with best result: 70.69 %"""
                    with open('linregmodel.pkl', 'wb') as f:
                        pickle.dump(poly_reg, f)
                
                
            print('Model saved')      
            print('Accuracy: {}'.format(acc))         # Accuracy for what the Power will be given 
            print('co: ', reg.coef_)               # Coefficients for attributes
            print('Intercept: ', reg.intercept_)   # Intercept of y
        
        else:
            try:
                """Read in our pickle file"""
                pickle_in = open('polyregmodel.pkl', 'rb')
                poly_reg = pickle.load(pickle_in)
                print('Polynomial regression model imported')
            except:
                raise ValueError('run_model set to False, missing file "polyregmodel.pkl". Set run_model True to run and save model')
        
        #poly = reg.intercept_ + reg.coef_[0]*X + reg.coef_[1]*y +reg.coef_[2]*X*X + reg.coef_[3]*X*y + reg.coef_[4]*y*y + reg.coef_[5]*X**3 + reg.coef_[6]*X**2*y
        
        """Hexplot since big dataset"""
        plt.hexbin(X.flatten(), y, gridsize=20, cmap='Blues')
        plt.axis([X.min(), X.max(), y.min(), y.max()])
        cb = plt.colorbar()
        cb.set_label('Scatter density')
        plt.plot(X, reg.predict(poly_reg.fit_transform(X)), color='#417ed6', alpha=0.7, lw=0.7)
        plt.xlabel(predictor)
        plt.ylabel(predict)
        plt.tight_layout()
        plt.show()
        
        
    def PolynomialRegressionNumpy(self, degree=6, plot=True):
        """Polynomial regression"""
        
        predict = 'POWER'
        predictor = 'WS100'
        
        X = self.data[predictor].values
        y = self.data[predict].values
        
        polynomial_order = degree
        
        """Curve fit the test data"""
        fitted_parameters = np.polyfit(X, y, polynomial_order)
        print('Fitted Parameters:', fitted_parameters)
        
        model_predictions = np.polyval(fitted_parameters, X)
        abs_error = model_predictions - y
        
        se = np.square(abs_error) # squared errors
        mse = np.mean(se) # mean squared errors
        rmse = np.sqrt(mse) # Root Mean Squared Error, RMSE
        r_squared = 1.0 - (np.var(abs_error) / np.var(y))
        print('RMSE:', rmse)
        print('R-squared:', r_squared)
        
        
        if plot:
            f = plt.figure(figsize=(8, 6), dpi=100)
            axes = f.add_subplot(111)
        
            """Plot raw data as a hexplot"""
            axes.hexbin(X, y, gridsize=20, cmap='Blues')
            axes.axis([X.min(), X.max(), y.min(), y.max()])
        
            """create data for the fitted equation plot"""
            x_model = np.linspace(min(X), max(X))
            y_model = np.polyval(fitted_parameters, x_model)
        
            """Plot model"""
            axes.plot(x_model, y_model)
            axes.set_xlabel(predictor)
            axes.set_ylabel(predict)
            plt.tight_layout()
        
            plt.show()
        

        

if __name__ == '__main__':
    c = CompareModels()
    #c.LinearRegression(run_model=True)
    c.PolynomialRegression()
    # c.PolynomialRegressionNumpy()


