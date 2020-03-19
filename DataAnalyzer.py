#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 00:44:02 2020

@author: alfredo
"""

import pandas as pd
import numpy as np


class Analyzer:
    def __init__(self):
        try:
            self.data = pd.read_pickle('data_by_mean.pkl')
        except:
            raise ImportError("Missing file: 'data_by_mean.pkl")
            
        self.data = self.data#[['POWER', 'WS10', 'WS100']]
        self.columns = self.data.columns

    def calc_wind_farm_direction(self):
        return
    
    def calc_roughness_coef(self):
        h = [10, 100]
        c = self.data.filter(['WS10', 'WS100'])
        a = np.mean(np.log(c['WS100'] / c['WS10']) * np.log(h[1] / h[0]))
        z_0 = np.exp(1/a)**-1
        print(z_0)
        
        
if __name__ == '__main__':
    obj = Analyzer()
    obj.calc_roughness_coef()
        