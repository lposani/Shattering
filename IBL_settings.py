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
    'cross_validations': 20,
    'ndata': 500
}

# ---------------- SD parameters ----------------

IBL_params = {
    'perf_thr': 0.666,
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

# ------ hierarchy settings ---------


VIS_areas = ['VISp', 'VISl', 'VISli', 'VISrl', 'VISpl', 'VISal', 'VISpor', 'VISa', 'VISam', 'VISpm']
SS_areas = ['SSp-ll', 'SSp-m', 'SSp-tr', 'SSp-ul', 'SSs', 'SSp-n', 'SSp-bfd', 'SSp-un']
F_areas = ['ACAd', 'ACAv', 'ORBl', 'ORBvl', 'ORBv', 'PL', 'ILA', 'ORBm', 'FRP']
M_areas = ['MOs', 'MOp']
ASS_areas = ['RSPagl', 'RSPd', 'RSPv', 'AId', 'AIv', 'AIp', 'GU', 'VISC', 'TEa', 'ECT', 'PERI']
AUD_areas = ['AUDp', 'AUDpo', 'AUDv', 'AUDd']

hierarchy = {
    0: VIS_areas + SS_areas + AUD_areas,
    1: ASS_areas,
    2: F_areas + M_areas,
}

H = {}
for key in hierarchy:
    for v in hierarchy[key]:
        H[v] = key

groups = {
    0: SS_areas,
    1: VIS_areas + AUD_areas,
    4: ASS_areas,  # Association
    2: F_areas,  # cognitive
    3: M_areas
}

colors = {}
for key in groups:
    for v in groups[key]:
        colors[v] = key

hierarchy_GPT = {
    1: ['VISp', 'AUDp', 'SSp-ll', 'SSp-m', 'SSp-tr', 'SSp-ul', 'SSp-n', 'SSp-bfd', 'SSp-un'],
    2: ['VISl', 'VISli', 'VISrl', 'VISpl', 'VISal', 'VISpor', 'VISa', 'VISam', 'VISpm', 'AUDv', 'AUDpo', 'AUDv', 'GU', 'SSs'],
    3: ['ACAd', 'ACAv', 'RSPagl', 'RSPd', 'RSPv', 'AId', 'AIv', 'AIp', 'GU', 'VISC', 'TEa', 'ECT', 'PERI', 'FRP', 'MOp', 'MOs'],
    4: ['PL', 'ILA', 'ORBl', 'ORBvl']
}
