from IBL_settings import *
from source.analysis.Dimensionality import *
from source.analysis.Clustering import *
folder = 'SD-v-clustering'
mkdir('./plots/IBL/SD-v-clustering')
mkdir('./plots/IBL/SD-v-clustering/SD')

reduced_CT = pickle.load(open('./datasets/IBL/reduced_CT.pck', 'rb'))
region = 'MOp'

# ---------- Find Clusters ----------
conditioned_trials = reduced_CT[region]
conditioned_trials_z = zscore_CT(conditioned_trials)
np.random.seed(0)
ss_data, mega_sus_mask, cluster_labels, X = classic_kmeans(conditioned_trials_z)

# Region-specific adjustments
n_neurons = IBL_params['neurons']
ndata_tot = decoding_params['ndata']
decoding_parhash = parhash(decoding_params)
nreps = 100 # None
trials = reduced_CT[region]
IC = len(trials[0].keys())
keys = list(trials[0].keys())
n = np.sum([c[keys[0]].shape[1] for c in trials])
megapooling = ceil(n_neurons / n)
print(region, n, megapooling, IC)
decoding_params['ndata'] = int(ndata_tot / IC)

# ---------- Find centroids per cluster
centroids = {c: np.nanmean(X[:, cluster_labels == c], 1) for c in np.unique(cluster_labels)}
for k in centroids:
    centroids[k] = np.asarray(megapooling*list(centroids[k]))

# Adapt to megapooling numbers
conditioned_trials = megapooling*conditioned_trials
cluster_labels = np.asarray(megapooling*list(cluster_labels))

PRs = []
SDs = []
ICs = []
SSs = []
alps = []

alphas = np.linspace(0, 1, 11)[:-1]

f = plt.figure(figsize=(8, 7))
g = GridSpec(20, 28, figure=f)

axs = [
    f.add_subplot(g[2:9, 4:14]),
    f.add_subplot(g[:13, 15:28], projection='3d'),
    f.add_subplot(g[11:18, 2:9]),
    f.add_subplot(g[11:18, 11:18]),
    f.add_subplot(g[11:18, 20:27])
    # f.add_subplot(g[12:18, 21:28]),
]

plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)
axs[1].set_xticklabels([])
axs[1].set_yticklabels([])
axs[1].set_zticklabels([])

axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].set_xlabel('PC 1')
axs[0].set_ylabel('PC 2')

decoding_params['cross_validations'] = 10
labels_str = [''.join(k) for k in conditioned_trials[0]]
colorbar = 0

xc = []
yc = []

x0s = []
y0s = []
z0s = []
pca = None


def plot_spaces(a):
    global x0s, y0s, z0s, pca, colorbar, xc, yc
    print(a)
    for ax in axs:
        ax.cla()

    axs[0].set_title('Conditions Space', color=pltcolors[3])
    axs[1].set_title('Activity Space', color=pltcolors[0])
    axs[2].set_title('Silhouette Score', color=pltcolors[3])
    axs[3].set_title('Participation Ratio', color=pltcolors[0])
    axs[4].set_title('Shattering Dimensionality')
    axs[2].set_xlabel('Cluster Diversity')
    axs[3].set_xlabel('Cluster Diversity')
    axs[4].set_xlabel('Cluster Diversity')

    alps.append(1-a)
    c_conditioned_trials = collapse_trials(conditioned_trials, cluster_labels, alpha=a)
    Xnew = CT_to_X(c_conditioned_trials, zscore=True)

    # visualize
    visualize_clusters(Xnew, cluster_labels, 'PCA', ax=axs[0])
    pca_trained = visualize_geometry(Xnew.T, condition_labels=labels_str, ax=axs[1], pcs=[1, 2, 3], fitted_pca=pca,
                                     equalize=True)
    PRs.append(participation_ratio(Xnew))

    # Shattering dim
    decoding_params['cluster_labels'] = cluster_labels
    decoding_params['collapse_alpha'] = a
    cache_name = f'{region}_a={a}_{parhash(decoding_params)}'
    print(cache_name)
    perfs, perfs_null, fingerprints = shattering_dimensionality(conditioned_trials,
                                                                nreps=nreps,
                                                                nnulls=2,
                                                                n_neurons=None,
                                                                region=region + 'a=%.2f' % a,
                                                                folder=folder,
                                                                IC=True,
                                                                cache_name=cache_name,
                                                                **decoding_params)

    SDs.append(np.nanmean(np.asarray(perfs) > 0.666))
    SSs.append(silhouette_score(Xnew.T, cluster_labels))

    axs[2].plot(alps, SSs, color=pltcolors[3], marker='.')
    axs[2].axvline(alps[-1], color='y', alpha=0.25)
    axs[2].set_xlim([-0.05, 1.05])
    axs[2].set_ylim([0.0, 1.02])

    axs[3].plot(alps, PRs, color=pltcolors[0], marker='.')
    axs[3].set_xlim([-0.05, 1.05])
    axs[3].set_ylim([0, 6.0])
    axs[3].axvline(alps[-1], color='y', alpha=0.25)

    axs[4].plot(alps, SDs, color=pltcolors[0], marker='.')
    axs[4].set_xlim([-0.05, 1.05])
    axs[4].set_ylim([0., 1.05])
    axs[4].axvline(alps[-1], color='y', alpha=0.25)
    linenull(axs[4], 0.0)
    axs[4].set_title('Separability')

    if a == alphas[0]:
        pca = pca_trained
        x0s = axs[1].get_xlim()
        y0s = axs[1].get_ylim()
        z0s = axs[1].get_zlim()
        xc = axs[0].get_xlim()
        yc = axs[0].get_ylim()

    axs[1].set_xlim(x0s)
    axs[1].set_ylim(y0s)
    axs[1].set_zlim(z0s)

    axs[0].set_xlim(xc)
    axs[0].set_ylim(yc)
    f.savefig(f'./plots/IBL/{folder}/SD/{region}_to_{a}.pdf')


# Set up the figure and axes
from matplotlib.animation import FuncAnimation, PillowWriter

anim = FuncAnimation(f, plot_spaces, frames=alphas, repeat=False)
anim.save(f'./plots/IBL/{folder}/{region}.gif', writer=PillowWriter(fps=5))

