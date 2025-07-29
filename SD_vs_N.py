import matplotlib.pyplot as plt
import numpy as np

from IBL_settings import *
from source.analysis.Dimensionality import *
from source.analysis.Clustering import *
import warnings
import sys

warnings.filterwarnings('ignore')
np.random.seed(0)

folder = 'SD-vs-N'

mkdir('./plots/IBL/' + folder + '/SD')
write_params('./plots/IBL/%s' % folder, IBL_params)

# Additional stuff
reduced_CT = pickle.load(open('./datasets/IBL/reduced_CT.pck', 'rb'))
cortical_regions = list(reduced_CT.keys())

ndata_tot = decoding_params['ndata']
decoding_parhash = parhash(decoding_params)
nnulls = 100
Ns = [60, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


def run(region, L=None):
    trials = reduced_CT[region]
    IC = len(trials[0].keys())
    keys = list(trials[0].keys())
    n = np.sum([c[keys[0]].shape[1] for c in trials])
    decoding_params['ndata'] = int(ndata_tot / IC)
    f, axs = plt.subplots(3, 1, figsize=(4, 8), gridspec_kw={'height_ratios': [4, 1.75, 1.75]})

    mean_perfs = []
    AUCS = []
    seps = []

    for n_neurons in Ns:
        if IC > 11:
            nreps = 500
        else:
            nreps = None

        megapooling = ceil(n_neurons / n)
        print(region, n, megapooling, IC)
        if L is not None:
            cache_name = f'{region}_IC_{decoding_parhash}_N={n_neurons}_L={L}'
            perfs_L, perfs_null_L, fingerprints = shattering_dimensionality(megapooling * reduced_CT[region],
                                                                            nreps=nreps,
                                                                            nnulls=100,
                                                                            n_neurons=n_neurons,
                                                                            region=region + f'_N={n_neurons}_L={L}',
                                                                            folder=folder,
                                                                            cache_name=cache_name,
                                                                            IC=True,
                                                                            subspace=L,
                                                                            **decoding_params)
            ts, y, ax = csd_plot(perfs_L, t0=np.percentile(perfs_null_L, 99), t1=np.nanmax(perfs_L),
                                 label=f'N={n_neurons}',
                                 ax=axs[0], linestyle='--')
        else:
            cache_name = f'{region}_IC_{decoding_parhash}_N={n_neurons}'
            perfs_L, perfs_null_L, fingerprints = shattering_dimensionality(megapooling * reduced_CT[region],
                                                                            nreps=nreps,
                                                                            nnulls=100,
                                                                            n_neurons=n_neurons,
                                                                            region=region + f'_N={n_neurons}',
                                                                            folder=folder,
                                                                            cache_name=cache_name,
                                                                            IC=True,
                                                                            **decoding_params)
            ts, y, ax = csd_plot(perfs_L, t0=np.percentile(perfs_null_L, 99), t1=np.nanmax(perfs_L),
                                 label=f'N={n_neurons}',
                                 ax=axs[0], linestyle='--')

        AUCS.append(np.nanmean(y))
        mean_perfs.append(np.nanmean(perfs_L))
        seps.append(np.nanmean(np.asarray(perfs_L) > np.nanmax(perfs_null_L)))

    ax = axs[0]
    ax.legend(fontsize=8)
    ax.set_xlabel('Decoding Performance')
    ax.set_xticks([0, 1.0])
    ax.set_xticklabels(['chance', '1.0'])
    ax.set_ylabel('CDF')

    ax = axs[1]
    ax.plot(Ns, AUCS, color='k', marker='o')
    ax.set_ylabel('perf cdf AUC')
    ax.set_xscale('log')
    ax.set_xticks(Ns)
    ax.set_xticklabels(Ns)
    ax.set_xlabel('N neurons')

    ax = axs[2]
    ax.plot(Ns, mean_perfs, color='k', marker='d')
    ax.set_ylabel('Mean Performance')
    ax.set_xscale('log')
    ax.set_xticks(Ns)
    ax.set_xticklabels(Ns)
    ax.set_xlabel('N neurons')
    ax.set_ylim([0.5, 1.0])
    if L is None:
        f.savefig(f'./plots/IBL/{folder}/{region}_Ns.pdf')
    else:
        f.savefig(f'./plots/IBL/{folder}/{region}_Ns_L={L}.pdf')
    plt.close(f)
    return np.asarray(mean_perfs), np.asarray(AUCS), np.asarray(seps)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python SD-pipeline.py <region>")
        sys.exit(1)
    region = sys.argv[1]
    if len(sys.argv) >= 3:
        L = int(sys.argv[2])
    else:
        L = None
    run(region, L)
