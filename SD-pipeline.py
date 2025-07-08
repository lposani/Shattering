from IBL_settings import *
from source.analysis.Dimensionality import *
from source.analysis.Clustering import *
import warnings

warnings.filterwarnings('ignore')
np.random.seed(0)

folder = 'SD IC cdf'
n_megapooled = IBL_params['neurons']

write_params('./plots/IBL/%s' % folder, IBL_params)

# Additional stuff
reduced_CT = pickle.load(open('./datasets/reduced_CT.pck', 'rb'))
cortical_regions = list(cortical_regions.values)

# Revised results

results_revision = {
    'region': [],
    'SD IC (alldics)': [],
    'perfs IC (alldics)': [],
    'perfs null IC (alldics)': []
}
decoding_params['cross_validations'] = 20
decoding_params['break_correlations'] = False
nnulls = 200

# Low-dimensional synthetic data
perfs_synthetic = {}
perfs_null_synthetic = {}

# Synthetic:
alpha = 0.0
r = 1.0
N = 1000
T = 200
sigma = 1.0

# for L in [2, 3, 4]:
#     decoding_params['ndata'] = int(400 / 2 ** L)
#     trials = generate_latent_representations(L=L, alpha=alpha, sigma=sigma, T=T, N=N, visualize=False)
#     perfs, perfs_null, fingerptints = shattering_dimensionality([trials], nreps=None,
#                                                                 region=f'Synthetic_L={L}_r={r}_N={N}_T={T}_s={sigma:.2f}_a={alpha:.2f}',
#                                                                 folder=folder,
#                                                                 nnulls=100,
#                                                                 cache_name=f'synthetic_L={L}_r={r}_N={N}_T={T}_s={sigma:.2f}_a={alpha:.2f}_{parhash(decoding_params)[1]}',
#                                                                 convert_dic=False,
#                                                                 **decoding_params)
#     perfs_synthetic[L] = perfs
#     perfs_null_synthetic[L] = perfs_null

for region in list(reduced_CT.keys())[::-1]:
    print(region)
    n_neurons = 4000  # IBL_params['neurons']

    IC = len(reduced_CT[region][0].keys())
    keys = list(reduced_CT[region][0].keys())
    decoding_params['ndata'] = int(500 / IC)
    n = np.sum([c[keys[0]].shape[1] for c in reduced_CT[region]])
    megapooling = ceil(n_neurons / n)
    print(n, megapooling, IC)

    # ---------- Shattering Dimensionality IC ----------
    print("Running: SD IC")
    perfs_region = {}

    if IC % 2:  # correct the dichotomies bug that only happened for odd IC
        cache_name = f'{region}_IC_{parhash(decoding_params)[1]}_corrected'
    else:
        cache_name = f'{region}_IC_{parhash(decoding_params)[1]}'
    perfs_orig, perfs_null, fingerprints = shattering_dimensionality(megapooling * reduced_CT[region],
                                                                     nreps=None,
                                                                     nnulls=100,
                                                                     n_neurons=n_neurons,
                                                                     region=region + '4000',
                                                                     folder=folder,
                                                                     cache_name=cache_name + '4000',
                                                                     IC=True,
                                                                     **decoding_params)

    f, axs = plt.subplots(2, 1, figsize=(4, 6), gridspec_kw={'height_ratios': [4, 1.75]})
    deltas = []

    ax = axs[0]
    ax.set_xlabel('Decoding Performance (normalized)')
    ax.set_ylabel('Cumulative Density')
    ts, y_orig, ax = csd_plot(perfs_orig, t0=np.percentile(perfs_null, 99), t1=np.nanmax(perfs_orig), label=region, ax=ax, linewidth=2)

    Ls = range(1, min(IC+1, 6))
    for L in Ls:
        trials = generate_latent_representations(L=L, P=IC, alpha=0, sigma=sigma, T=T, N=N, visualize=False)
        cache_name = f'synthetic_IC_{parhash(decoding_params)[1]}_P={IC}_L={L}_s={sigma}'
        perfs_L, perfs_null_L, fingerprints_L = shattering_dimensionality([trials],
                                                                          nreps=None,
                                                                          nnulls=100,
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
