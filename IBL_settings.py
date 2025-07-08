import pandas as pd
import numpy as np

# ---------------- Data loading ----------------
IBL_meta = pd.read_csv('./datasets/IBL/IBL_regions.csv')
cosmos = pd.unique(IBL_meta['Cosmos'])
# shuqi_path = './datasets/shuqi_uuids_may30.csv'
shuqi_path = './datasets/uuids_Aug2.csv'
cortical_regions = IBL_meta['Beryl'][IBL_meta['Cosmos'] == 'Isocortex']

H = pd.read_csv('./datasets/IBL/area_list.csv', header=None).to_dict()[0]
inverted_H = {H[key]: key for key in H}
inverted_H['SSp'] = -1

# ---------------- Conditioned trials parameters ----------------
variables = {
    'Whisking': {
        'W': lambda d: d['whisking_left'] > np.percentile(d['whisking_left'], 50),
        'S': lambda d: d['whisking_left'] < np.percentile(d['whisking_left'], 50)
    },
    'Context': {
        '2': lambda d: (d['context'] == 0.2),
        '8': lambda d: (d['context'] == 0.8)
    },
    # 'Movement': {
    #     'm': lambda d: (d['time_from_firstmove'] > 0),
    #     's': lambda d: (d['time_from_firstmove'] < 0),
    # },
    'Stimulus': {
        '<': lambda d: (d['position'] == 1),
        '>': lambda d: (d['position'] == -1)
    },
    'Contrast': {
        'l': lambda d: (d['contrast'] < 0.2),
        'h': lambda d: d['contrast'] > 0.2,
    },
}

# ---------------- Decoding parameters ----------------
decoding_params = {
    'training_fraction': 0.8,
    'nshuffles': 0,
    'cross_validations': 100,
    'ndata': 100
}

# ---------------- SD parameters ----------------

IBL_params = {
    'perf_thr': 0.666,
    'SDreps': 200,
    'neurons': 640,
    'min_trials': 5,
    't0': 0,
    't1': 1.0,
    'vars': '-'.join(list(variables.keys())),
    'uuid': shuqi_path
}

filters = [
    lambda s: (s['time_from_onset'] > IBL_params['t0'])
              & (s['time_from_onset'] < IBL_params['t1'])
]

