# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:41:43 2020

@author: Sigve Sørensen
"""

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import matplotlib
# matplotlib.use('Agg')
%matplotlib inline

import datetime as dt, itertools, pandas as pd, matplotlib.pyplot as plt, numpy as np

import utility as util

global logger

class ForecastPower:
    """Forecast out T+6h soloely based on historical Power data"""
    
    def __init__(self):
        try:
            self.data = pd.read_pickle('data_by_mean.pkl')
        except:
            raise ImportError("Missing file: 'data_by_mean.pkl")
            
        #self.data = self.data[['POWER', 'WS10', 'WS100', 'U10', 'U100']]
        self.columns = self.data.columns




util.setup_log()
util.setup_path()
logger = util.logger
text_process.logger = logger

use_cuda = torch.cuda.is_available()
logger.info("Is CUDA available? %s.", use_cuda)
