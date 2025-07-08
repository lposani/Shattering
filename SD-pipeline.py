from IBL_settings import *
from source.analysis.Dimensionality import *
from source.analysis.Clustering import *
import warnings

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
cortical_regions = list(cortical_regions.values)

# Revised results

results_revision = {
    'region': [],
    'SD IC (alldics)': [],
    'perfs IC (alldics)': [],
    'perfs null IC (alldics)': []
}
decoding_params['cross_validations'] = 20
nnulls = 100

for region in ['SSp-n']: # list(reduced_CT.keys())[::-1]:
    print(region)
    n_neurons = IBL_params['neurons']

    IC = len(reduced_CT[region][0].keys())
    keys = list(reduced_CT[region][0].keys())
    decoding_params['ndata'] = int(500 / IC)
    n = np.sum([c[keys[0]].shape[1] for c in reduced_CT[region]])
    megapooling = ceil(n_neurons / n)
    print(n, megapooling, IC)

    # ---------- Shattering Dimensionality IC ----------
    print("Running: SD IC")
    perfs_region = {}

    cache_name = f'{region}_IC_{parhash(decoding_params)}'
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

    Ls = range(1, min(IC+1, 7))
    for L in Ls:
        trials = generate_latent_representations(L=L, P=IC, **synthetic_params)
        cache_name = f'synthetic_IC_{parhash(decoding_params)}_P={IC}_L={L}_{parhash(synthetic_params)}'
        perfs_L, perfs_null_L, fingerprints_L = shattering_dimensionality([trials],
                                                                          nreps=None,
                                                                          nnulls=10,
                                                                          region=f'synthetic_P={IC}_L={L}',
                                                                          folder=folder,
                                                                          cache_name=cache_name,
                                                                          IC=True,
                                                                          **decoding_params)

        ts, y, ax = csd_plot(perfs_L, t0=np.percentile(perfs_null_L, 99), t1=np.nanmax(perfs_L), label=f'Synthetic L={L} P={IC}',
                             ax=ax, linestyle='--')
        deltas.append(np.nanmean(y - y_orig))
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

    results_revision['region'].append(region)
    results_revision['SD IC (alldics)'].append(np.nanmean(perfs_orig > np.percentile(perfs_null, 99)))
    results_revision['perfs IC (alldics)'].append(perfs_orig)
    results_revision['perfs null IC (alldics)'].append(perfs_null)

    pickle.dump(results_revision, open('./datasets/IBL_revisions_SD.pck', 'wb'))
