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
import statsmodels.api as sm

from math import sqrt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor



class CompareModels:
    """Compare several regression models for the best model fit"""
    
    def __init__(self):
        
        try:
            self.data = pd.read_pickle('data_by_mean.pkl')
        except:
            raise ImportError("Missing file: 'data_by_mean.pkl")
            
        self.data = self.data[['POWER', 'WS10', 'WS100', 'U10', 'U100']]
        self.columns = self.data.columns
        
        
    def scatterplot_winds(self):
        """Compare power distribution for the two wind-speeds"""
        
        f = plt.figure(figsize=(8, 6), dpi=100)
        ax = f.add_subplot(111)
        
        x1 = self.data['WS10']
        x2 = self.data['WS100']
        y = self.data['POWER']
        
        ax.scatter(x1, y, color='b', alpha=0.3)
        ax.scatter(x2, y, color='g', alpha=0.1)
        
        ax.axis([x1.min(), x2.max(), y.min(), y.max()])
        plt.xlabel('WS10, WS100')
        plt.ylabel('POWER')
        plt.title('Comparing wind-speeds at heights 10m and 100m')
        plt.tight_layout()
        plt.show()    
        
        
    def LinearRegression(self, model=True, runs=1500):
        """Perform a linear regression"""
            
        target = 'POWER'    # Target variable
        feature = 'WS100'   # Feature variable
        
        X = self.data[feature].values.reshape(-1,1)
        y = self.data[target].values
        
        best_acc = 0
        
        """Looping to find the best fit linear model"""
        if model:
            for _ in range(runs):
            
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                linear = LinearRegression()
                linear.fit(X_train, y_train)
                
                y_pred = linear.predict(X_test)
                acc = r2_score(y_test, y_pred)
                rmse = sqrt(mean_squared_error(y_test, y_pred))
                nrmse = rmse / y.mean()
                
                if acc > best_acc:
                    print('Model Accuracy: {} %'.format(round(acc*100,3)))
                    best_acc = acc
                    
                    """Save our model with the best accuracy"""
                    with open('linregmodel.pkl', 'wb') as handle:
                        pickle.dump(linear, handle)
                        
            print('----------Model saved----------')       
            print('R-squared: {}'.format(best_acc))   # Accuracy for what the Power will be given
            print('RMSE: {}'.format(rmse))            # Root Mean Square Error (RMSE)
            print('NRMSE: {}'.format(nrmse))          # Normalized Root Mean Square Error (NRMSE)
            print('Coeff: ', linear.coef_)            # Coefficients for attributes (a)
            print('Intercept: ', linear.intercept_)   # Intercept of y (b)
            
        else:
            try:
                """Read in our pickle file"""
                pickle_in = open('linregmodel.pkl', 'rb')
                linear = pickle.load(pickle_in)
                print('Linear regression model imported')
            except:
                raise ValueError('run_model set to False, missing file "linregmodel.pkl". Set run_model True to run and save model.')
        
        """Plotting the best fit linear regression line"""
        f = plt.figure(figsize=(8, 6), dpi=100)
        ax = f.add_subplot(111)
        
        ax.hexbin(X, y, gridsize=28, cmap='Blues', norm=mlt.colors.LogNorm()) # Plotting data distribution
        ax.axis([X.min(), X.max(), y.min(), y.max()])
        pcm = ax.get_children()[0]
        cb = plt.colorbar(pcm, ax=ax,)
        cb.set_label('Data point density', fontsize=13)
        
        ax.plot(X_test, y_pred, color='k', alpha=0.4, lw=3) # Linear Regression line
        plt.xlabel(feature, fontsize=13)
        plt.ylabel(target, fontsize=13)
        plt.title('Linear Regression Fit, Acc = {} %'.format(round(best_acc*100, 3)), fontsize=15)
        plt.tight_layout()
        plt.show()       
        
        
    def MultipleRegression(self, pair_plot=False, model=True, runs=1500):
        """Perform a multiple regression"""
        
        
        X = self.data.drop(['POWER'], axis=1)
        y = self.data['POWER']
        
        """
        Variance inflation factor VIF
        In turn removing any feature with VIF > 30, dropping them above
        Getting rid of multicolinearity, ind. var. that highly relate to each other
        """
        vif = pd.DataFrame()
        vif['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['features'] = X.columns
        print(vif.round())
        
        X = self.data.drop(['POWER', 'WS10', 'U10'], axis=1)
        
        if pair_plot:
            sb.pairplot(self.data, x_vars=['WS100', 'U100'], y_vars=['POWER'], plot_kws={"s": 3})
            plt.show()
            
        """
        Backward elimination:
        POWER=b0+b1*"WS100"+b2*"U100"
        POWER=b0*1+b1*"WS100"+b2*"U100"
        Need to add one more feature, a constant 1
        """
        
        X=sm.add_constant(X)
        #print(X.head())
        regressorOLS = sm.OLS(y, X).fit()
        print(regressorOLS.summary())
        
        """
        No P>|t| greater than significance level 0.05. Keep features
        Can now create our model:
        """
        
        best_acc = 0
        
        """Looping to find the best fit linear model"""
        if model:
            for _ in range(runs):
            
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                linear = LinearRegression()
                linear.fit(X_train, y_train)
                
                y_pred = linear.predict(X_test)
                acc = r2_score(y_test, y_pred)
                rmse = sqrt(mean_squared_error(y_test, y_pred))
                nrmse = rmse / y.mean()
                
                if acc > best_acc:
                    print('Model Accuracy: {} %'.format(round(acc*100,3)))
                    best_acc = acc
                    
                    """Save our model with the best accuracy"""
                    with open('multipleregmodel.pkl', 'wb') as handle:
                        pickle.dump(linear, handle)
                        
            print('----------Model saved----------')       
            print('R-squared: {}'.format(best_acc))   # Accuracy for what the Power will be given
            print('RMSE: {}'.format(rmse))            # Root Mean Square Error (RMSE)
            print('NRMSE: {}'.format(nrmse))          # Normalized Root Mean Square Error (NRMSE)
            print('Coeff: ', linear.coef_)            # Coefficients for attributes (a)
            print('Intercept: ', linear.intercept_)   # Intercept of y (b)
            
        else:
            try:
                """Read in our pickle file"""
                pickle_in = open('multipleregmodel.pkl', 'rb')
                linear = pickle.load(pickle_in)
                print('Multiple regression model imported')
            except:
                raise ValueError('run_model set to False, missing file "linregmodel.pkl". Set run_model True to run and save model.')


        """Plot distribution actual- vs. predicted power"""        
        sb.distplot(y_test, color='#0080FF', label='Actual Values',
                    hist=True, bins = 25,
                    norm_hist=True).set_title('Actual- vs. Predicted Power Output, Acc = {} %'.format(round(best_acc*100, 3)))
        sb.distplot(y_pred, color='#FB6805', label='Predicted Values',
                    hist=True, norm_hist=True, bins=25)
        plt.xlabel('Power (W)', fontsize=13)
        plt.ylabel('Probability Density', fontsize=13)
        plt.legend()
        plt.show()
        
        
    def PolynomialRegression(self, model=True, runs=1000, plot=True):
        """Perform a polynomial regression"""
        
        target = 'POWER'    # Target variable
        feature = 'WS100'   # Feature variable
        
        self.data = self.data.sort_values(by=[feature])
        
        X = self.data[feature].values.reshape(-1,1)
        y = self.data[target].values
        
        best_acc = 0 # Store best accuracy
        
        """Looping to find the best fit polynomial regression model"""
        if model:
            for _ in range(runs):
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                """
                y = b0*1 + b1*X + b2*X^2 + ... + bd*X^d
                where d = polynomial degree
                """
                poly = PolynomialFeatures(degree = 6)
                X_poly = poly.fit_transform(X) # Expands the equation according to poly degree
                
                poly.fit(X_poly, y) # Fitting model
                linear = LinearRegression()
                linear.fit(X_poly, y)
                
                y_poly_pred = linear.predict(X_poly)
                
                """Compare actual- to predicted values"""
                # tmp = pd.DataFrame({'Actual': y, 'Predicted': y_poly_pred})
                # print(tmp.head())
                
                acc = r2_score(y, y_poly_pred)
                rmse = sqrt(mean_squared_error(y, y_poly_pred))
                nrmse = rmse / y.mean()
                
                if acc > best_acc:
                    print(acc)
                    best_acc = acc
                    
                    """Save our model with pickle"""
                    with open('polyregmodel.pkl', 'wb') as handle:
                        pickle.dump(poly, handle)
                
            print('----------Model saved----------')      
            print('R-squared: {}'.format(best_acc))   # Accuracy for what the Power will be given
            print('RMSE: {}'.format(rmse))            # Root Mean Square Error (RMSE)
            print('NRMSE: {}'.format(nrmse))          # Normalized Root Mean Square Error (NRMSE)
            print('co: ', linear.coef_)               # Coefficients for attributes
            print('Intercept: ', linear.intercept_)   # Intercept of y
        
        else:
            try:
                """Read in our pickle file"""
                pickle_in = open('polyregmodel.pkl', 'rb')
                poly = pickle.load(pickle_in)
                print('Polynomial regression model imported')
            except:
                raise ValueError('run_model set to False, missing file "polyregmodel.pkl". Set run_model True to run and save model')
        
        if plot:
            """Plot """
            f = plt.figure(figsize=(8, 6), dpi=100)
            ax = f.add_subplot(111)
            
            ax.hexbin(X[:,0], y, gridsize=28, cmap='Blues', norm=mlt.colors.LogNorm()) # Plotting data distribution
            ax.axis([X.min(), X.max(), y.min(), y.max()])
            pcm = ax.get_children()[0]
            cb = plt.colorbar(pcm, ax=ax,)
            cb.set_label('Data point density', fontsize=13)
            
            ax.plot(X, y_poly_pred, color='k', alpha=0.4, lw=3) # Linear Regression line
            plt.xlabel(feature, fontsize=13)
            plt.ylabel(target, fontsize=13)
            plt.title('Polynomial Regression Fit, Acc = {} %'.format(round(best_acc*100, 3)), fontsize=15)
            plt.tight_layout()
            plt.show()  
        
        
    def PolynomialRegressionNumpy(self, degree=6, plot=True, model=True):
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
        
        """Model evaluation"""
        se = np.square(abs_error) # squared errors
        mse = np.mean(se) # mean squared errors
        rmse = np.sqrt(mse) # Root Mean Squared Error, RMSE
        r_squared = 1.0 - (np.var(abs_error) / np.var(y))
        print('RMSE:', rmse)
        print('R-squared:', r_squared)
        
        x_model = np.linspace(min(X), max(X))
        y_model = np.polyval(fitted_parameters, x_model)
        
        
        if plot:
            
            f = plt.figure(figsize=(8, 6), dpi=100)
            ax = f.add_subplot(111)
            
            ax.hexbin(X, y, gridsize=28, cmap='PuBu', norm=mlt.colors.LogNorm())
            ax.axis([X.min(), X.max(), y.min(), y.max()])
            pcm = ax.get_children()[0]
            cb = plt.colorbar(pcm, ax=ax)
            cb.set_label('Data point density')
            ax.plot(x_model, y_model, color='k', alpha=0.4, lw=4)
            plt.xlabel=predictor
            plt.ylabel=predict
            plt.title('Polynomial Regression, Polynomial order {}'.format(degree))
            plt.tight_layout()
            plt.show() 
            
            
    
    def kNNeighborsRegression(self, plot=True, fit_model=True, det_k=True, plot_k=True):
        """kNN Regression"""
        
        
        if fit_model: 
            
            predict = "POWER"
            predictor = "WS100"
            
            X = self.data[predictor].values.reshape(-1,1)
            #y = self.data.drop([predict], 1).values
            y = self.data[predict].values
            
            best_acc = 0    # Store best accuracy
            runs = 100      # Number of loops to find best fit
            n_neigh = 37    # How many points are considered as neighbors
            
            if det_k:
                """Determine the optimal number of neighbors"""
                lowest_rmse = 1
                store_rmse = []
                for k in range(100):
                    k = k + 1
                    knn = KNeighborsRegressor(k)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                    knn.fit(X_train, y_train)
                    y_pred_knn = knn.predict(X_test)
                    rmse = sqrt(mean_squared_error(y_test, knn.predict(X_test)))
                    store_rmse.append(rmse)
                    if rmse < lowest_rmse:
                        lowest_rmse = rmse
                        n_neigh = k
                print('n neighbors set to {}'.format(n_neigh))
                
                if plot_k:
                    """Display the RMSE as a function of k"""
                    plt.figure(figsize=(10,8))
                    k_plot = pd.DataFrame(store_rmse)
                    k_plot.plot()
                    plt.xlabel("k neighbors")
                    plt.ylabel("RMSE")
                    plt.show()
                    
            
            knn = KNeighborsRegressor(n_neighbors=n_neigh) # Model
            
            """Run the kNN fitting n times (runs) to find the best fit and save model"""
            for _ in range(runs):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                #kNN model fit:
                knn.fit(X_train, y_train)
                
                #Predict:
                y_pred_knn = knn.predict(X_test)
                
                acc = r2_score(y_test, knn.predict(X_test))
                rmse = sqrt(mean_squared_error(y_test, knn.predict(X_test)))
                
                if acc > best_acc:
                    best_acc = acc
                    
                    """Save our model with the best accuracy"""
                    with open('knnmodel.pkl', 'wb') as handle:
                        pickle.dump(knn, handle)
                        
                        
            print('Model saved')       
            print('Accuracy: {}'.format(best_acc))  
            print('k: {}'.format(n_neigh))        
            print('Root Mean Squared Error: {}'.format(rmse))
        else:
            try:
                """Read in our pickled model"""
                pickle_in = open('knnmodel.pkl', 'rb')
                knn = pickle.load(pickle_in)
                print('kNN model imported')
            except:
                raise ValueError('fit_model set to False, missing file "knnmodel.pkl". Set run_model True to fit model')
        

        
        if plot:
            """Display kNN regression"""
            f = plt.figure(figsize=(8, 6), dpi=100)
            ax = f.add_subplot(111)
            ax.hexbin(X_test[:,0], y_pred_knn, gridsize=28, cmap='PuBu', norm=mlt.colors.LogNorm())
            ax.axis([X.min(), X.max(), y.min(), y.max()])
            pcm = ax.get_children()[0]
            cb = plt.colorbar(pcm, ax=ax)
            cb.set_label('Data point density')
            plt.xlabel(predictor)
            #plt.xlabel(str(self.data.columns.drop([predict]).values))
            plt.ylabel(predict)
            plt.title('kNN Regression, k = {}'.format(n_neigh))
            plt.tight_layout()
            plt.show() 
       
        
        
    def SupportVectorRegression(self, plot=True, model=True):
        
        from sklearn.svm import SVR
        import matplotlib
        
        predict = 'POWER'
        predictor = 'WS100'
        
        if model:
            
            X = self.data[predictor][:500].values.reshape(-1,1)
            y = self.data[predict][:500].values
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)   
            
            
            svr_rbf = SVR(kernel='rbf', C=1e3, gamma = 0.1)
            #svr_lin = SVR(kernel='linear', C=1e3)
            #svr_poly = SVR(kernel='poly', C=1e3, degree=2)
            y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
            #y_lin = svr_lin.fit(X, y).predict(X)
            #y_poly = svr_poly.fit(X, y).predict(X)
            
            # linear = LinearRegression()
            # acc = linear.score(X, y)
            # print(acc)
        else:
            pass
            
        if plot:  
            
            
            f = plt.figure(figsize=(8, 6), dpi=100)
            ax = f.add_subplot(111)
            
            ax.hexbin(X.flatten(), y, gridsize=28, cmap='Blues', norm=matplotlib.colors.LogNorm())
            ax.axis([X.min(), X.max(), y.min(), y.max()])
            pcm = ax.get_children()[0]
            cb = plt.colorbar(pcm, ax=ax)
            cb.set_label('Data point density')
            ax.plot(X, y_rbf, color='k', alpha=0.4, lw=4)
            plt.xlabel(predictor)
            plt.ylabel(predict)
            plt.title('Support Vector Regression')
            plt.tight_layout()
            plt.show() 
            
            
        #     plt.show()
        #     plt.scatter(X, y, color='k', label=predictor)
        #     plt.plot(X, y_rbf, color='r', lw=3, label='RBF Model')
        #     #plt.plot(X, y_lin, color='c', lw=lw, linestyle='-.', label='Linear Model')
        #     #plt.plot(X, y_poly, color='navy', lw=lw, linestyle='--', label='Polynomial Model')
        #     plt.xlabel(predictor)
        #     plt.ylabel(predict)
        #     plt.title('Support Vector Regression')
        #     plt.legend()
        #     plt.show()
        # else:
        #    pass
            
            
            
        #     for _ in range(1000):
                
        #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
            
        #         clf = svm.SVC()
        #         clf.fit(X_train, y_train)
                
        #         best_acc = 0
                
        #         acc = clf.score(X_test, y_test)
                
        #         if acc > best_acc:
        #             print(acc)
        #             best_acc = acc
                    
        #             """Save our model with pickle. Saved with best result"""
        #             with open('svm-model.pkl', 'wb') as f:
        #                 pickle.dump(clf, f)
                    
        # else:
        #     try:
        #         """Read pickled SVM model"""
        #         pickle_in = open('svm-model.pkl', 'rb')
        #         clf = pickle.load(pickle_in)
        #         print('Polynomial regression model imported')
        #     except:
        #         raise ValueError('run_model set to False, missing file "svm-model.pkl", set run_model True')
                
        

if __name__ == '__main__':
    c = CompareModels()
    
    #c.LinearRegression()
    c.MultipleRegression()
    #c.PolynomialRegression()
    #c.PolynomialRegressionNumpy(degree=6)
    #c.kNNeighborsRegression()
    #c.SupportVectorRegression()
    #c.scatterplot_winds()
    #c.LinearRegression(run_model=True)