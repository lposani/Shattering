from source import *
from source.analysis.Dimensionality import *
from source.analysis.Clustering import *
from IBL_settings import *

folder = 'simulation'
mkdir('./plots/IBL/simulation/SD')
import sys
decoding_params['cross_validations'] = 10
decoding_params['ndata'] = 25


def generate_clustered_data(N, P, k, diversity, sigma, trial_per_condition=100, seed=None):
    # Generate random centroids with unitary module and centered around zero
    if seed is not None:
        np.random.seed(seed)
    centroids = np.random.randn(k, P)
    centroids = np.linalg.qr(centroids.T)[0].T
    centroids = centroids - np.nanmean(centroids, 0)
    for i in range(k):
        centroids[i] = centroids[i] / scipy.linalg.norm(centroids[i])

    # Create N-dim representations by repeating N/k times the centroid neuron
    Ss = [np.repeat(c, int(N / k)).reshape(P, int(N / k)).T for c in centroids]
    X = np.vstack(Ss).T

    # Add diversity (quenched noise) and create the NxP matrix
    X += diversity * np.random.randn(P, N) / np.sqrt(P)

    # Now add trial to trial variability
    trials = {
        f'c{i:02d}': X[i][np.newaxis, :] + np.random.randn(trial_per_condition, N) * sigma for i in range(P)
    }

    return trials


def plot_simulation(N, k, sigma, P, alphas):
    PRs = []
    SDs = []
    DPs = []
    SSs = []
    alps = []

    # getting limits: 0
    a = alphas[0]
    conditioned_trials = generate_clustered_data(N=N, P=P, k=k, diversity=a, sigma=sigma, seed=0)
    Xnew = CT_to_X(conditioned_trials, zscore=False)
    PR0 = participation_ratio(Xnew)
    cache_name = f'synthetic_a={a:.2f}_k={k}_N={N}_P={P}_s={sigma}_{parhash(decoding_params)}'
    perfs, perfs_null, fingerprints = shattering_dimensionality([conditioned_trials],
                                                                nreps=100,
                                                                nnulls=100,
                                                                n_neurons=None,
                                                                region='simulated_' + 'a=%.2f' % a,
                                                                folder=folder,
                                                                IC=True,
                                                                cache_name=cache_name,
                                                                **decoding_params)
    SD0 = np.nanmean(np.asarray(perfs) > np.nanmax(perfs_null))
    DP0 = np.nanmean(perfs)
    cluster_labels = np.repeat(np.arange(k), int(N / k))
    SS0 = silhouette_score(Xnew.T, cluster_labels)
    # ---

    # getting limits: 1
    a = alphas[-1]
    conditioned_trials = generate_clustered_data(N=N, P=P, k=k, diversity=a, sigma=sigma, seed=0)
    Xnew = CT_to_X(conditioned_trials, zscore=False)
    PR1 = participation_ratio(Xnew)
    cache_name = f'synthetic_a={a:.2f}_k={k}_N={N}_P={P}_s={sigma}_{parhash(decoding_params)}'
    perfs, perfs_null, fingerprints = shattering_dimensionality([conditioned_trials],
                                                                nreps=100,
                                                                nnulls=100,
                                                                n_neurons=None,
                                                                region='simulated_' + 'a=%.2f' % a,
                                                                folder=folder,
                                                                IC=True,
                                                                cache_name=cache_name,
                                                                **decoding_params)
    SD1 = np.nanmean(np.asarray(perfs) > np.nanmax(perfs_null))
    DP1 = np.nanmean(perfs)
    cluster_labels = np.repeat(np.arange(k), int(N / k))
    SS1 = silhouette_score(Xnew.T, cluster_labels)
    # ---

    f = plt.figure(figsize=(12, 9))
    g = GridSpec(20, 36, figure=f)

    axs = [
        f.add_subplot(g[2:9, 4:14]),
        f.add_subplot(g[:13, 15:28], projection='3d'),
        f.add_subplot(g[12:18, 2:9]),
        f.add_subplot(g[12:18, 11:18]),
        f.add_subplot(g[12:18, 20:27]),
        f.add_subplot(g[12:18, 29:36]),
    ]

    plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)
    axs[1].set_xticklabels([])
    axs[1].set_yticklabels([])
    axs[1].set_zticklabels([])

    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_xlabel('PC 1')
    axs[0].set_ylabel('PC 2')

    colorbar = 0

    xc = []
    yc = []

    x0s = []
    y0s = []
    z0s = []
    pca = visualize_geometry(Xnew.T, condition_labels=list(conditioned_trials.keys()), ax=axs[1], pcs=[1, 2, 3],
                             fitted_pca=None, equalize=True, lims=None)
    # equalize_axs(axs[1])
    visualize_clusters(Xnew, cluster_labels, 'PCA', ax=axs[0])
    equalize_axs(axs[0])

    x0s = axs[1].get_xlim()
    y0s = axs[1].get_ylim()
    z0s = axs[1].get_zlim()
    xc = axs[0].get_xlim()
    yc = axs[0].get_ylim()
    ds = (SS0 - SS1)/20
    dp = (PR1 - PR0)/20
    dd = (DP1 - DP0)/20
    dsd = (SD1 - SD0)/20
    da = (alphas[-1]-alphas[0])/20

    def plot_spaces(a):
        nonlocal x0s, y0s, z0s, pca, colorbar, xc, yc
        print(a)
        for ax in axs:
            ax.cla()

        axs[0].set_title('Conditions Space', color=pltcolors[3])
        axs[1].set_title('Activity Space', color=pltcolors[0])
        axs[2].set_title('Silhouette Score', color=pltcolors[3])
        axs[3].set_title('Participation Ratio', color=pltcolors[0])
        axs[4].set_title('Separability')
        axs[5].set_title('Mean Dec Perf')

        axs[2].set_xlabel('Cluster Diversity')
        axs[3].set_xlabel('Cluster Diversity')
        axs[4].set_xlabel('Cluster Diversity')
        axs[5].set_xlabel('Cluster Diversity')

        alps.append(a)
        conditioned_trials = generate_clustered_data(N=N, P=P, k=k, diversity=a, sigma=sigma, seed=0)
        cluster_labels = np.repeat(np.arange(k), int(N / k))

        Xnew = CT_to_X(conditioned_trials, zscore=False)

        # visualize
        visualize_clusters(Xnew, cluster_labels, 'PCA', ax=axs[0])
        equalize_axs(axs[0])
        pca_trained = visualize_geometry(Xnew.T, condition_labels=list(conditioned_trials.keys()), ax=axs[1],
                                         pcs=[1, 2, 3], fitted_pca=pca,
                                         equalize=True, lims=[x0s, y0s, z0s])
        PRs.append(participation_ratio(Xnew))
        equalize_axs(axs[1])

        # Shattering dim
        cache_name = f'synthetic_a={a:.2f}_k={k}_N={N}_P={P}_s={sigma}_{parhash(decoding_params)}'
        print(cache_name)
        perfs, perfs_null, fingerprints = shattering_dimensionality([conditioned_trials],
                                                                    nreps=100,
                                                                    nnulls=100,
                                                                    n_neurons=None,
                                                                    region='simulated_' + 'a=%.2f' % a,
                                                                    folder=folder,
                                                                    IC=True,
                                                                    cache_name=cache_name,
                                                                    **decoding_params)

        SDs.append(np.nanmean(np.asarray(perfs) > np.nanmax(perfs_null)))
        DPs.append(np.nanmean(perfs))
        SSs.append(silhouette_score(Xnew.T, cluster_labels))

        axs[2].plot(alps, SSs, color=pltcolors[3], marker='.')
        axs[2].axvline(alps[-1], color='y', alpha=0.25)
        axs[2].set_xlim([alphas[0]-da, alphas[-1]+da])
        axs[2].set_ylim([SS1 - ds, SS0 + ds])

        axs[3].plot(alps, PRs, color=pltcolors[0], marker='.')
        axs[3].set_xlim([alphas[0]-da, alphas[-1]+da])
        axs[3].set_ylim([PR0-dp, PR1+dp])
        axs[3].axvline(alps[-1], color='y', alpha=0.25)

        axs[4].plot(alps, SDs, color=pltcolors[0], marker='.')
        axs[4].set_xlim([alphas[0]-da, alphas[-1]+da])
        axs[4].set_ylim([SD0-dsd, 1+dsd])
        axs[4].axvline(alps[-1], color='y', alpha=0.25)

        axs[5].plot(alps, DPs, color=pltcolors[0], marker='.')
        axs[5].set_xlim([alphas[0]-da, alphas[-1]+da])
        axs[5].set_ylim([0.45, 1.05])
        axs[5].axvline(alps[-1], color='y', alpha=0.25)
        linenull(axs[5], 0.5)

        axs[1].set_xlim(x0s)
        axs[1].set_ylim(y0s)
        axs[1].set_zlim(z0s)

        axs[0].set_xlim(xc)
        axs[0].set_ylim(yc)
        f.savefig(f'./plots/IBL/{folder}/SD/k={k}_N={N}_s={sigma}_to_{a:.2f}.pdf')

    # Set up the figure and axes
    from matplotlib.animation import FuncAnimation, PillowWriter

    anim = FuncAnimation(f, plot_spaces, frames=alphas, repeat=False)
    anim.save(f'./plots/IBL/{folder}/synthetic_k={k}_N={N}_s={sigma}.gif', writer=PillowWriter(fps=5))

    savename = f'./datasets/IBL/SD_cache/synthetic_k={k}_N={N}_P={P}_s={sigma}_a0={alphas[0]}_af={alphas[-1]}_na={len(alphas)}.pck'
    pickle.dump([PRs, SDs, DPs, SSs], open(savename, 'wb'))


if __name__ == '__main__':
    sigma = float(sys.argv[1])

    if len(sys.argv) > 3:
        N = int(sys.argv[2])
        k = int(sys.argv[3])
    else:
        N = 640
        k = 2

    P = 16
    alphas = np.linspace(0.05, 2.0, 40)
    plot_simulation(N, k, sigma, P=16, alphas=alphas)
