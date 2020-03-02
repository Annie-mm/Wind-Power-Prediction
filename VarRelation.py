import pandas as pd
import numpy as np
import DataCleaner

class VarRelation:
    def __init__(self):
        self.data = DataCleaner.clean_data(method='mean')
        
    
if __name__ == '__main__':
    self.data