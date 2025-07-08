import copy
import os.path
import pickle
import time
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import seaborn as sns
from numpy.random import RandomState
from matplotlib import gridspec
import hashlib
import scipy
from sklearn.base import BaseEstimator
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import csv
from pathlib import Path

from decodanda import *

cmap = cm.get_cmap('cool', 256)
pltcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
pltcolors = pltcolors * 100
