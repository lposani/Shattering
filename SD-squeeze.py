from IBL_settings import *
from source.analysis.Dimensionality import *
from source.analysis.Clustering import *
import warnings
import sys

warnings.filterwarnings('ignore')
np.random.seed(0)

folder = 'IC-squeeze'

mkdir('./plots/IBL/' + folder + '/SD')
write_params('./plots/IBL/%s' % folder, IBL_params)

# Additional stuff
reduced_CT = pickle.load(open('./datasets/IBL/reduced_CT.pck', 'rb'))
cortical_regions = list(reduced_CT.keys())

n_neurons = IBL_params['neurons']
ndata_tot = decoding_params['ndata']
decoding_parhash = parhash(decoding_params)
nnulls = 100


def run(region):
    trials = reduced_CT[region]
    IC = len(trials[0].keys())
    keys = list(trials[0].keys())
    n = np.sum([c[keys[0]].shape[1] for c in trials])
    megapooling = ceil(n_neurons / n)
    print(region, n, megapooling, IC)
    decoding_params['ndata'] = int(ndata_tot / IC)

    cache_name = f'{region}_IC_{decoding_parhash}'
    perfs_orig, perfs_null, fingerprints = shattering_dimensionality(megapooling * reduced_CT[region],
                                                                     nreps=None,
                                                                     nnulls=nnulls,
                                                                     n_neurons=n_neurons,
                                                                     region=region,
                                                                     folder=folder,
                                                                     cache_name=cache_name,
                                                                     IC=True,
                                                                     **decoding_params)

    f, axs = plt.subplots(2, 1, figsize=(4, 6), gridspec_kw={'height_ratios': [4, 1.75]})
    ax = axs[0]

    deltas = []
    ts, y_orig, ax = csd_plot(perfs_orig, t0=np.percentile(perfs_null, 99), t1=np.nanmax(perfs_orig), label=region, ax=ax,
                              linewidth=2)
    ax.set_xlabel('Decoding Performance (normalized)')
    ax.set_ylabel('Cumulative Density')

    Ls = np.arange(1, IC, dtype=int)
    for L in Ls:
        if IC > 11:
            nreps = 500
        else:
            nreps = None
        cache_name = f'{region}_IC_{decoding_parhash}_collapse_{L}'
        perfs_L, perfs_null_L, fingerprints = shattering_dimensionality(megapooling * reduced_CT[region],
                                                                        nreps=nreps,
                                                                        nnulls=100,
                                                                        n_neurons=None,
                                                                        region=region + f'_collapsed_{L}',
                                                                        folder=folder,
                                                                        cache_name=cache_name,
                                                                        IC=True,
                                                                        subspace=L,
                                                                        **decoding_params)
        ts, y, ax = csd_plot(perfs_L, t0=np.percentile(perfs_null_L, 99), t1=np.nanmax(perfs_L),
                             label=f'Projected L={L}',
                             ax=ax, linestyle='--')
        deltas.append(np.nanmean(y - y_orig))

    ax.legend(fontsize=8)
    ax = axs[1]
    ax.plot(Ls, deltas, color='k')
    for i, L in enumerate(Ls):
        plt.plot(L, deltas[i], color=pltcolors[i + 1], marker='o')
    ax.axhline(0, color=pltcolors[0], linestyle='--', linewidth=2.0)
    ax.set_ylabel('$\Delta$ AUC')
    ax.set_xticks(Ls)
    ax.set_xlabel('Latent dimensionality $L$')
    f.savefig(f'./plots/IBL/{folder}/{region}_cdf.pdf')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python SD-pipeline.py <region>")
        sys.exit(1)

    region = sys.argv[1]
    run(region)
