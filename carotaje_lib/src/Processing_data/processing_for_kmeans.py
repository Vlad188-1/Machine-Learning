# My impotrs ===============================
import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')
import configargparse
from tqdm import tqdm
# ==========================================


def process(data_1):
    #random.seed(10)
    def engine(x):       
        all_wells = []
        for i in x['well id'].unique():
            all_wells.append(x[x['well id']==i])
            
        for i in tqdm(range(len(all_wells)), desc='Preprocessing files...'):            
            all_wells[i]['diff_depth'] = all_wells[i]['depth, m'].diff(periods=-1)
            all_wells[i] = all_wells[i].loc[~all_wells[i]['diff_depth'].isin([0.0])]
            all_wells[i] = all_wells[i].dropna()
            all_wells[i] = all_wells[i].drop(['GZ2','GZ3','GZ4','GZ5', 'NKTM', 'NKTR', 'NKTD'], axis = 1)
            all_wells[i]['diff_ALPS'] = all_wells[i]['ALPS'].diff(periods = -1).abs()
            all_wells[i]['ALPS*BK'] = all_wells[i]['ALPS']*all_wells[i]['bk']
            all_wells[i]['bk**2'] = pow(all_wells[i]['bk'],2)
            all_wells[i]['smooth_diff_ALPS'] = all_wells[i]['diff_ALPS'].rolling(window=4, min_periods=1).mean()
            all_wells[i] = all_wells[i].drop(['diff_depth'], axis=1)
            all_wells[i] = all_wells[i].dropna()
        frame = pd.concat(all_wells, axis=0) 
        return frame  

    all_wells_1_process = engine(data_1)
    #id_test_wells = random.sample(list(all_wells_1_process['well id'].unique()), 5)
    test_wells = [all_wells_1_process[all_wells_1_process['well id']==i] for i in [26,14,75, 23, 44]]
    all_wells_1_process = all_wells_1_process.loc[~all_wells_1_process['well id'].isin([26,14,75, 23, 44])]
    x_train = all_wells_1_process.drop(['well id', 'depth, m', 'lith', 'goal'], axis=1)
    y_train = all_wells_1_process['goal']
    
    return x_train, y_train, test_wells, all_wells_1_process
