# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:30:32 2020

@author: Sigve Sørensen
"""

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import warnings
import itertools
import pandas as pd
plt.style.use('fivethirtyeight')

class ForecastPower:
    """
    Univariate time-series forecasting of WT output power
    
    SARIMAX(p,d,q)(P,D,Q)s
    p - Auto-Regressive (AR) , P - Seasonal component of p
    d - Integrated (I) , D - Seasonal component of d
    q - Moving Average (MA) - Seasonal component of q
    s - period
    """
    
    def __init__(self):
        
        # Import data from pickle
        try:
            self.data = pd.read_pickle('data_by_mean.pkl')
        except:
            raise ImportError("Missing file: 'data_by_mean.pkl")
            
        # Time series data    
        self.data = self.data['POWER']
        
        # SARIMAX(0, 0, 1)x(0, 1, 0, 24)24 - AIC:143.27089144512297
        self.mod = sm.tsa.statespace.SARIMAX(self.data,
                                             order=(0, 0, 1),
                                             seasonal_order=(0, 1, 0, 24),
                                             enforce_stationarity=False,
                                             enforce_invertibility=False)
        
        self.results = self.mod.fit()
        
    def optimize_model(self):
        """
        Optimize parameter order for SARIMAX
        SARIMAX(p,d,q)(P,D,Q)s
        p - Auto-Regressive (AR) , P - Seasonal component of p
        d - Integrated (I) , D - Seasonal component of d
        q - Moving Average (MA) - Seasonal component of q
        s - period
        """
        
        # Define the p, d and q parameter range for optimization
        p = d = q = range(0, 2)
        
        # Generate combinations of p, q and q
        pdq = list(itertools.product(p, d, q))
        
        # Generate combinations of seasonal P, D and Q
        pdq_s = [(x[0], x[1], x[2], 24) for x in pdq]
        
        print('Examples of parameter combinations for SARIMAX...')
        print(f'SARIMAX: {pdq[1]} x {pdq_s[1]}')
        print(f'SARIMAX: {pdq[1]} x {pdq_s[2]}')
        print(f'SARIMAX: {pdq[2]} x {pdq_s[3]}')
        print(f'SARIMAX: {pdq[3]} x {pdq_s[4]}')
        
        warnings.filterwarnings("ignore") # specify to ignore warning messages

        for order in pdq:
            for order_s in pdq_s:
                try:
                    mod = sm.tsa.statespace.SARIMAX(self.data,
                                                order=order,
                                                seasonal_order=order_s,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
        
                    results = mod.fit()
        
                    print(f'{order}x{order_s} - AIC:{results.aic}')
                except:
                    continue
                
    
    def plot_diagnostics(self):
        """Display diagnostics and statistical summary"""
        
        # Statistical summary
        print(self.results.summary().tables[1])
        
        # Diagnostics
        self.results.plot_diagnostics(figsize=(15, 12))
        plt.show()
                
        
    def SARIMAX_forecast(self, dynamic=True):
        """
        SARIMAX 'one-step-ahead' time series forecast
        For static forecast set dynamic=False
        """
       
        if dynamic:
            """Dynamic 'one-step-ahead' forecast"""
            
            # Set t0 for x-axis
            self.start_x_axis = self.data['2013-11-24 00:00:00':]
            # Set t0 for forecast
            self.start_forecast=self.data['2013-11-24 12:00:00':]
            
            # Forecast
            self.pred_dynamic = self.results.get_prediction(
                start=pd.to_datetime('2013-11-24 12:00:00'),
                dynamic=True, full_results=True)
            
            # Extract confidence intervals
            self.pred_dynamic_ci = self.pred_dynamic.conf_int()
            
            # Extract the predicted and true values of our time series
            self.forecasted = self.pred_dynamic.predicted_mean
            self.truth = self.start_forecast
            
            # Compute the mean square error
            self.mse = ((self.forecasted - self.truth) ** 2).mean()
            self.rmse = np.sqrt(self.mse)
            print(f'MSE of dynamic forecasts: {round(self.mse, 2)}')
            print(f'RMSE of dynamic forecasts is {round(self.rmse, 2)}')
            
            # Plot result
            ax = self.start_x_axis.plot(label='Observed',
                                        figsize=(20, 15),
                                        fontsize=22, lw=5)
            self.pred_dynamic.predicted_mean.plot(label='Predicted', 
                                                  ax=ax, lw=5)
            
            ax.fill_between(self.pred_dynamic_ci.index,
                            self.pred_dynamic_ci.iloc[:, 0],
                            self.pred_dynamic_ci.iloc[:, 1],
                            color='k', alpha=.25)
            
            ax.fill_betweenx(ax.get_ylim(), 
                             pd.to_datetime('2013-11-24 12:00:00'),
                             self.data.index[-1], alpha=.1, zorder=-1)
            
            ax.set_xlabel('Date', fontsize=35)
            ax.set_ylabel('POWER', fontsize=35)
            anchored_text = AnchoredText(f"RMSE: {round(self.rmse, 3)}", loc=2,
                                         prop=dict(size=35), frameon=True)
            ax.add_artist(anchored_text)
            plt.title("'one-step-ahead' SARIMAX dynamic forecast of 'POWER'",
                      fontsize=35)
            ax.legend(fontsize=40, loc=3)
            plt.tight_layout()
            plt.show()
            
        else:
            """Static 'one-step-ahead' forecast"""
            
            # Set t0 for x-axis
            self.start_x_axis = self.data['2013-11-24 00:00:00':]
            # Set t0 for forecast
            self.start_forecast=self.data['2013-11-24 12:00:00':]
            
            # Forecast
            self.pred = self.results.get_prediction(
                start=pd.to_datetime('2013-11-24 12:00:00'), dynamic=False
                )
            
            # Get confidence intervals of forecast
            self.pred_ci = self.pred.conf_int() # Confidence interval
            
            # Extract the predicted and true values of our time series
            self.forecasted = self.pred.predicted_mean
            self.truth = self.start_forecast
            
            # Compute the mean square error
            self.mse = ((self.forecasted - self.truth) ** 2).mean()
            self.rmse = np.sqrt(self.mse)
            print(f'MSE of dynamic forecasts: {round(self.mse, 2)}')
            print(f'RMSE of dynamic forecasts is {round(self.rmse, 2)}')
            
            # Plot result
            ax = self.start_x_axis.plot(label='Observed',
                                        figsize=(20, 15),
                                        fontsize=25, lw=5)
            self.pred.predicted_mean.plot(ax=ax, alpha=.7,
                                     label='Predicted', lw=5)
            
            ax.fill_between(self.pred_ci.index,
                            self.pred_ci.iloc[:, 0],
                            self.pred_ci.iloc[:, 1], color='k', alpha=.2)
            anchored_text = AnchoredText(f"RMSE: {round(self.rmse, 3)}", loc=2,
                                         prop=dict(size=30), frameon=True)
            ax.add_artist(anchored_text)
            ax.set_xlabel('Date', fontsize=35)
            ax.set_ylabel('POWER', fontsize=35)
            plt.title("'one-step-ahead' SARIMAX forecast of 'POWER'",
                      fontsize=35)
            ax.legend(fontsize=40, loc=4)
            plt.tight_layout()
            plt.show()
            
        
    def SARIMAX_forecast_out(self, STEPS=6):
        """Forecast into the future"""
        
        # Get forecast 6h (steps) ahead in future
        self.pred_uc = self.results.get_forecast(steps=STEPS)
        
        # Get confidence intervals of forecast
        self.pred_ci = self.pred_uc.conf_int()
        
        # Plot results
        ax = self.data['2013-11-29 00:00:00':].plot(label='Observed',
                                                    figsize=(20, 15),
                                                    fontsize=25, lw=7)
        self.pred_uc.predicted_mean.plot(ax=ax, label='Forecast', lw=7)
        ax.fill_between(self.pred_ci.index,
                        self.pred_ci.iloc[:, 0],
                        self.pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Date', fontsize=35)
        ax.set_ylabel('POWER', fontsize=35)
        plt.title(f"{STEPS}h ahead SARIMAX forecast of 'POWER'",
                  fontsize=35)
        ax.legend(fontsize=40, loc=3)
        plt.tight_layout()
        plt.show()
        

if __name__ == "__main__":
    
    tmp = ForecastPower()
    #tmp.optimize_model()
    tmp.plot_diagnostics()
    #tmp.SARIMAX_forecast(dynamic=False)
    tmp.SARIMAX_forecast_out(STEPS=6)
    
        
        
        
        
        
        
        
        
        
        
        
        