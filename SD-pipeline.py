from IBL_settings import *
from source.analysis.Dimensionality import *
from source.analysis.Clustering import *
import warnings
import sys

warnings.filterwarnings('ignore')
np.random.seed(0)

folder = 'IC'

synthetic_params = {
    'N': 400,
    'T': 100,
    'sigma': 0.5,
    'alpha': 0
}

mkdir('./plots/IBL/'+folder+'/SD')
mkdir('./datasets/IBL/SD_cache')
write_params('./plots/IBL/%s' % folder, IBL_params)
write_params('./plots/IBL/%s' % folder, synthetic_params, name='synthetic_params')

# Additional stuff
reduced_CT = pickle.load(open('./datasets/IBL/reduced_CT.pck', 'rb'))
cortical_regions = list(reduced_CT.keys())

# Revised results

ndata_tot = decoding_params['ndata']
nnulls = 100
decoding_parhash = parhash(decoding_params)


def run(region):
    n_neurons = IBL_params['neurons']
    IC = len(reduced_CT[region][0].keys())
    keys = list(reduced_CT[region][0].keys())
    n = np.sum([c[keys[0]].shape[1] for c in reduced_CT[region]])
    megapooling = ceil(n_neurons / n)
    print(region, n, megapooling, IC)

    decoding_params['ndata'] = int(ndata_tot / IC)

    # ---------- Shattering Dimensionality IC ----------
    print("Running: SD IC")
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
    deltas = []

    ax = axs[0]
    ax.set_xlabel('Decoding Performance (normalized)')
    ax.set_ylabel('Cumulative Density')
    ts, y_orig, ax = csd_plot(perfs_orig, t0=np.percentile(perfs_null, 99), t1=np.nanmax(perfs_orig), label=region, ax=ax, linewidth=2)
    Ls = range(1, IC+1)
    for L in Ls:
        synthetic_params['sigma'] = 1.0
        # Tune sigma in synthetic_params
        sigma = tune_noise(region=region, trials=megapooling*reduced_CT[region], L=L, P=IC,
                           synthetic_params=synthetic_params, decoding_params=decoding_params, IBL_params=IBL_params)

        synthetic_params['sigma'] = sigma

        # Run
        trials = generate_latent_representations(L=L, P=IC, **synthetic_params)
        cache_name = f'synthetic_IC_{decoding_parhash}_P={IC}_L={L}_{parhash(synthetic_params)}'
        perfs_L, perfs_null_L, fingerprints_L = shattering_dimensionality([trials],
                                                                          nreps=None,
                                                                          nnulls=10,
                                                                          region=f'synthetic/P={IC}_L={L}_s={synthetic_params["sigma"]}',
                                                                          folder=folder,
                                                                          cache_name=cache_name,
                                                                          IC=True,
                                                                          **decoding_params)

        ts, y, ax = csd_plot(perfs_L, t0=np.percentile(perfs_null_L, 99), t1=np.nanmax(perfs_L), label=f'Synthetic L={L} P={IC}',
                             ax=ax, linestyle='--')
        deltas.append(np.nanmean(y - y_orig))
        if deltas[-1] < 0:
            break

    ax.legend(fontsize=8)
    ax = axs[1]
    ax.plot(Ls, deltas, color='k')
    for i, L in enumerate(Ls):
        plt.plot(L, deltas[i], color=pltcolors[i+1], marker='o')
    ax.axhline(0, color=pltcolors[0], linestyle='--', linewidth=2.0)
    ax.set_ylabel('$\Delta$ AUC')
    ax.set_xlabel('Latent dimensionality $L$')
    f.savefig(f'./plots/IBL/{folder}/{region}_cdf.pdf')
    plt.close(f)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python SD-pipeline.py <region>")
        sys.exit(1)

    region = sys.argv[1]
    run(region)
