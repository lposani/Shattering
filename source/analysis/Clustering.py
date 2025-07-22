import numpy as np
import os, pickle
import pandas as pd
import scipy.spatial.distance
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from source.utilities import sort_matrix, rotate_rasters, visualize_data_vs_null
import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group
from source.utilities import *
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.gridspec import GridSpec


class kmeans_analyzer(object):
    def __init__(self, k=None, max_k=None):
        self.k = k
        self.max_k = max_k

    def __call__(self, X, **kwargs):
        D = X  # np.corrcoef(X)
        # for i in range(len(D)):
        #     D[i][np.isnan(D[i])] = 0
        if self.k is None:
            if self.max_k is None:
                self.max_k = np.min([20, X.shape[0] - 1])
            silhouettes = []
            ks = np.arange(2, self.max_k)
            for k in ks:
                # print(k)
                clusterer = KMeans(n_clusters=k, random_state=10, n_init='auto')
                cluster_labels = clusterer.fit_predict(D)
                silhouette_avg = silhouette_score(D, cluster_labels)
                silhouettes.append(silhouette_avg)

            k = ks[np.argmax(silhouettes)]
            # print(f'Best k={k}')
            # print(silhouettes)
        else:
            k = self.k
        clusterer = KMeans(n_clusters=k, random_state=10, n_init='auto')
        cluster_labels = clusterer.fit_predict(D)
        silhouette = silhouette_score(D, cluster_labels)
        silhouette_values = silhouette_samples(D, cluster_labels)

        # print(f'silhouette: {silhouette}')
        idx = np.argsort(cluster_labels)
        D_sorted = D[idx, :]

        # plt.subplots()
        # plt.pcolor(D)
        # plt.subplots()
        # plt.pcolor(D_sorted)
        return silhouette, D_sorted, k, cluster_labels, silhouette_values


def CT_to_SID(conditioned_trials):
    sid = []
    for i in range(len(conditioned_trials)):
        sid.append(np.ones(list(conditioned_trials[i].values())[0].shape[1]) * i)
    return np.hstack(sid)


def visualize_X(conditioned_trials, find_clusters=True):
    keys = [''.join(k) for k in conditioned_trials[0].keys()]
    km = kmeans_analyzer()
    X = CT_to_X(conditioned_trials)
    nanmask = np.isnan(np.sum(X, 0)) == 0
    X = X[:, nanmask]
    C = np.corrcoef(X.T)
    D, idx = sort_matrix(C)
    if find_clusters:
        res = km(X.T)
        cluster_labels = res[3]
        f, axs = plt.subplots(2, 2, figsize=(8, 12), gridspec_kw={'width_ratios': [5, 1.5]})
        idx_cl = np.argsort(cluster_labels)
        axs[0, 1].pcolor(X[:, idx].T, rasterized=True)
        axs[0, 0].pcolor(D, rasterized=True)
        axs[1, 1].pcolor(X[:, idx_cl].T, rasterized=True)
        axs[1, 0].pcolor(C[idx_cl, :][:, idx_cl], rasterized=True)
        if keys:
            axs[0, 1].set_xticks(np.arange(len(keys)) + 0.5)
            axs[0, 1].set_xticklabels(keys, rotation=90, fontsize=8)
            axs[1, 1].set_xticks(np.arange(len(keys)) + 0.5)
            axs[1, 1].set_xticklabels(keys, rotation=90, fontsize=8)
    else:
        f, axs = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [5, 1.5]})
        axs[1].pcolor(X[:, idx].T, rasterized=True)
        axs[0].pcolor(D, rasterized=True)
        if keys:
            axs[1].set_xticks(np.arange(len(keys)) + 0.5)
            axs[1].set_xticklabels(keys, rotation=90, fontsize=8)

    return f, axs, idx


def visualize_clusters(X, cluster_labels, method='PCA', ax=None, pcs=None, patches=False, dots=True, thresh=0.5):
    if pcs is None:
        pcs = [0, 1]

    numeric_cluster_labels = np.zeros(len(cluster_labels))
    for i, c in enumerate(np.unique(cluster_labels)):
        mask = cluster_labels == c
        numeric_cluster_labels[mask] = i

    # Assume X is your (P, N) matrix and cluster_labels is your label vector of size N
    # Transpose X to have shape (N, P) for PCA
    cluster_labels = np.array(cluster_labels)  # Ensure labels are a NumPy array

    if method == 'PCA':
        pca = PCA(n_components=4)
        X_reduced = pca.fit_transform(X.T)
    if method == 'tSNE':
        tsne = TSNE(n_components=4, perplexity=30, random_state=42, n_iter=1000)
        X_reduced = tsne.fit_transform(X.T)  # X_reduced has shape (N, 2)

    # Create the scatter plot
    if ax is None:
        f, ax = plt.subplots(figsize=(5, 5))

    colormap = plt.cm.tab20
    colors = colormap(np.linspace(0, 1, 20))

    if patches:
        index1 = pcs[0]
        index2 = pcs[1]
        for i, c in enumerate(np.unique(cluster_labels)):
            dict_data = {f'PC{pcs[0]}': X_reduced[cluster_labels == c, pcs[0]],
                         f'PC{pcs[1]}': X_reduced[cluster_labels == c, pcs[1]],
                         'label': cluster_labels[cluster_labels == c]
                         }
            sns.kdeplot(pd.DataFrame.from_dict(dict_data), x=f'PC{pcs[0]}', y=f'PC{pcs[1]}',
                        ax=ax, color=colors[i], fill=True, thresh=thresh, levels=2, legend=False, alpha=0.3)
            ax.plot([np.nanmedian(X_reduced[cluster_labels == c, pcs[0]], 0)],
                    [np.nanmedian(X_reduced[cluster_labels == c, pcs[1]], 0)],
                    'o', color=colors[i], label=c)

    if dots:
        for i, c in enumerate(np.unique(cluster_labels)):
            index1 = pcs[0]
            index2 = pcs[1]
            print(index1, type(index1))
            ax.scatter(
                X_reduced[cluster_labels == c, index1],
                X_reduced[cluster_labels == c, index2],
                c=colors[i],
                cmap='tab20',  # Use a categorical colormap
                alpha=0.75,
                label=c
            )
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.colorbar(scatter, label='Cluster Labels')  # Add a colorbar for reference
    if method == 'PCA':
        ax.set_xlabel(f'Principal Component {pcs[0] + 1}')
        ax.set_ylabel(f'Principal Component {pcs[1] + 1}')
    if method == 'tSNE':
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
    return ax


def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must be of the same length")
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))


def visualize_geometry(X, condition_labels, ax=None, pcs=None, fitted_pca=None, equalize=True):
    if fitted_pca is None:
        fitted_pca = PCA(n_components=4)
        X_reduced = fitted_pca.fit_transform(X.T)
    else:
        X_reduced = fitted_pca.transform(X.T)

    if pcs is None:
        pcs = [0, 1]
    if len(pcs) == 2:
        if ax is None:
            f, ax = plt.subplots(figsize=(4, 4))

        for i, ci in enumerate(condition_labels):
            ax.text(X_reduced[i, 0], X_reduced[i, 1], ci, color='k', fontsize=8)
            for j, cj in enumerate(condition_labels):
                if hamming_distance(ci, cj) == 1:
                    ax.plot([X_reduced[i, 0], X_reduced[j, 0]], [X_reduced[i, 1], X_reduced[j, 1]], '-o', color=pltcolors[0])
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
    if len(pcs) == 3:
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(projection='3d')
        for i in range(len(condition_labels)):
            ax.scatter(X_reduced[i, 0], X_reduced[i, 1], X_reduced[i, 2], alpha=0.7,
                       marker='$%s$' % condition_labels[i],
                       s=1000, color=pltcolors[i], edgecolor='w', linewidths=5)
            ax.scatter(X_reduced[i, 0], X_reduced[i, 1], X_reduced[i, 2], alpha=0.7,
                       marker='$%s$' % condition_labels[i],
                       s=1000, color=pltcolors[i])
        for i in range(len(condition_labels)):
            for j in range(i + 1, len(condition_labels)):
                if hamming_distance(condition_labels[i], condition_labels[j]) == 1:
                    ax.plot([X_reduced[i][0], X_reduced[j][0]], [X_reduced[i][1], X_reduced[j][1]],
                            [X_reduced[i][2], X_reduced[j][2]], linestyle='-', linewidth=2, color='k',
                            alpha=0.7)
        if equalize:
            equalize_ax(ax)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    return fitted_pca


def split_ct(conditioned_trials, training_fraction):
    conditioned_trials_training = []
    conditioned_trials_testing = []

    for trial in conditioned_trials:
        training_trial = {}
        testing_trial = {}

        for key, matrix in trial.items():
            num_rows = matrix.shape[0]
            num_train = int(num_rows * training_fraction)

            # Shuffle the rows before splitting
            indices = np.arange(num_rows)
            np.random.shuffle(indices)

            train_indices = indices[:num_train]
            test_indices = indices[num_train:]

            training_trial[key] = matrix[train_indices]
            testing_trial[key] = matrix[test_indices]

        conditioned_trials_training.append(training_trial)
        conditioned_trials_testing.append(testing_trial)

    return conditioned_trials_training, conditioned_trials_testing


def cv_kmeans(conditioned_trials, cross_validations=10, training_fraction=0.75, exclude_sus_clusters=False):
    km = kmeans_analyzer()
    ss_data = []
    for k in range(cross_validations):
        # divide conditioned trials into training and testing
        conditioned_trials_training, conditioned_trials_testing = split_ct(conditioned_trials, training_fraction)

        # convert training and testing conditioned trials into X matrices
        X_training = CT_to_X(conditioned_trials_training)
        X_testing = CT_to_X(conditioned_trials_testing)
        nanmask = (np.isnan(np.sum(X_training, 0)) == 0) & (np.isnan(np.sum(X_testing, 0)) == 0)
        X_training = X_training[:, nanmask]
        X_testing = X_testing[:, nanmask]

        session_id = CT_to_SID(conditioned_trials)
        session_id = session_id[nanmask]

        res = km(X_training.T)

        def find_sus_neurons(res):
            cl = res[3]
            sv = res[4]
            suspicious_mask = np.zeros(len(sv))
            for c in np.unique(cl):
                mask = cl == c
                # print(cl.shape, sv.shape, len(session_id))
                c_sv = sv[mask]
                c_sid = session_id[mask]
                for c_sid_i in np.unique(c_sid):
                    c_sid_i_mask = c_sid == c_sid_i
                    if np.sum(c_sv[c_sid_i_mask]) > 0.9 * np.sum(c_sv):
                        print("FOUND ONE: ", np.sum(c_sv[c_sid_i_mask]) / np.sum(c_sv))
                        suspicious_mask[(cl == c) & (session_id == c_sid_i)] = 1
            return suspicious_mask

        if exclude_sus_clusters:
            sus_mask = find_sus_neurons(res)
            while np.sum(sus_mask):
                print("FOUND A SUS CLUSTER!")
                print("old X shape is:", X_training.shape)
                X_training = X_training[:, sus_mask == 0]
                X_testing = X_testing[:, sus_mask == 0]
                session_id = session_id[sus_mask == 0]
                res = km(X_training.T)
                sus_mask = find_sus_neurons(res)
                print("new X shape is:", X_training.shape)

        cluster_labels = res[3]
        D = np.corrcoef(X_testing.T)
        silhouette_avg = silhouette_score(D, cluster_labels)
        ss_data.append(silhouette_avg)

    return ss_data


def sample_gaussian_data(X):
    n_dimensions, n_samples = X.shape
    null_data = np.random.multivariate_normal(np.nanmean(X, 1), np.cov(X), size=n_samples)
    return null_data.T


def classic_kmeans(conditioned_trials, exclude_sus_clusters=True, norm=False, gaussian=False):
    km = kmeans_analyzer()
    ss_data = []
    X = CT_to_X(conditioned_trials)
    if norm:
        X = X / np.sum(X, 0, keepdims=True)

    nanmask = (np.isnan(np.sum(X, 0)) == 0) & (np.isnan(np.sum(X, 0)) == 0)
    X = X[:, nanmask]

    session_id = CT_to_SID(conditioned_trials)
    session_id = session_id[nanmask]

    res = km(X.T)
    mega_sus_mask = np.ones(X.shape[1]).astype(bool)

    def find_sus_neurons(res):
        cl = res[3]
        sv = res[4]
        suspicious_mask = np.zeros(len(sv))
        for c in np.unique(cl):
            mask = cl == c
            # print(cl.shape, sv.shape, len(session_id))
            c_sv = sv[mask]
            c_sid = session_id[mask]
            for c_sid_i in np.unique(c_sid):
                c_sid_i_mask = c_sid == c_sid_i
                if np.sum(c_sv[c_sid_i_mask]) > 0.9 * np.sum(c_sv):
                    print("FOUND ONE: ", np.sum(c_sv[c_sid_i_mask]) / np.sum(c_sv))
                    suspicious_mask[(cl == c) & (session_id == c_sid_i)] = 1
        return suspicious_mask

    if exclude_sus_clusters:
        sus_mask = find_sus_neurons(res)
        mega_sus_mask = (mega_sus_mask) & (sus_mask == 0)
        while np.sum(sus_mask):
            print("FOUND A SUS CLUSTER!")
            print("old X shape is:", X.shape)
            X = X[:, sus_mask == 0]
            session_id = session_id[sus_mask == 0]
            res = km(X.T)
            sus_mask = find_sus_neurons(res)
            mega_sus_mask = (mega_sus_mask) & (sus_mask == 0)
            print("new X shape is:", X.shape)

    if gaussian:
        X = sample_gaussian_data(X)

    cluster_labels = res[3]
    D = np.corrcoef(X.T)
    silhouette_avg = silhouette_score(D, cluster_labels)
    ss_data = silhouette_avg

    return ss_data, mega_sus_mask, cluster_labels, X


def kmeans_analysis(conditioned_trials, nshuffles=25, cross_validations=10, training_fraction=0.75,
                    exclude_sus_clusters=False):
    ss_data = np.nanmean(cv_kmeans(conditioned_trials, cross_validations=cross_validations,
                                   training_fraction=training_fraction, exclude_sus_clusters=exclude_sus_clusters))

    null = []
    for i in tqdm(range(nshuffles)):
        rotated_ct = []
        for ct in conditioned_trials:
            n_neurons = list(ct.values())[0].shape[1]
            R = special_ortho_group.rvs(n_neurons)
            rotated_ct.append(rotate_rasters(ct, R))
        ss_n_k = cv_kmeans(rotated_ct, cross_validations=cross_validations,
                           training_fraction=training_fraction, exclude_sus_clusters=exclude_sus_clusters)
        null.append(np.nanmean(ss_n_k))

    return ss_data, null


def _get_best(ms_hist, _train_and_eval, **kwargs):
    _ = np.argmax(ms_hist['ms_metric'])
    best_param = ms_hist['param'][_]
    best_metric = ms_hist['ms_metric'][_]
    res = _train_and_eval(**kwargs, **best_param)
    return res


def remove_space(s):
    s = s.replace(" ", "")
    s = s.replace("'", "")
    s = s.replace("[", "")
    s = s.replace("]", "")
    s = s.replace("{", "")
    s = s.replace("}", "")
    s = s.replace(":", "")
    s = s.replace(",", "_")
    return s


def load_or_save_dict(_fname, _main, **params):
    if os.path.isfile(_fname):
        with open(_fname, 'rb') as f:
            res = pickle.load(f)
    else:
        res = _main(**params)
        with open(_fname, 'wb') as f:
            pickle.dump(res, f)
    return res


def kmeans_sort(data_mat, n_clus_lim=None, dis_metric='euclidean',
                ms_metric="sscore", n_init=10,
                save_prefix=None, **others):
    assert dis_metric == "euclidean", "kmeans only take euclidean distance"

    def _train_and_eval(**kwargs):
        clustering = KMeans(random_state=42,
                            n_init=kwargs['n_init'],
                            n_clusters=kwargs['n_clusters']).fit(data_mat)  # n_init=200
        labels = clustering.labels_
        sscore = silhouette_score(data_mat, labels)
        sscore_samples = silhouette_samples(data_mat, labels)
        sscore_macro = np.mean([np.mean(sscore_samples[labels == li]) for li in range(np.max(labels) + 1)])
        # if kwargs['ms_metric'] == "sscore":
        if ms_metric == "sscore":
            ms_score = sscore
        # elif kwargs['ms_metric'] == "sscore_macro":
        elif ms_metric == "sscore_macro":
            ms_score = sscore_macro
        else:
            assert False, "invalid ms_metric"
        return dict(eval_metrics={"sscore": sscore, "sscore_macro": sscore_macro},
                    ms_metric=ms_score, res=labels,
                    model=clustering, n_clus=len(np.unique(labels)),
                    **kwargs)

    def _main(n_clus_lim):
        ms_hist = {'param': [], "eval_metrics": [], "ms_metric": []}
        for n_clus in tqdm(range(n_clus_lim[0], n_clus_lim[1]), desc="kmeans_sort"):
            if n_clus >= len(data_mat): continue
            param = dict(n_clusters=n_clus, n_init=n_init)
            sc_clus_res = _train_and_eval(**param)
            ms_hist['param'].append(param)
            ms_hist['eval_metrics'].append(sc_clus_res['eval_metrics'])
            ms_hist['ms_metric'].append(sc_clus_res['ms_metric'])

        best_res = _get_best(ms_hist, _train_and_eval)
        best_res['ms_hist'] = ms_hist
        return best_res

    save_midfix = remove_space(f"{n_clus_lim}")
    if save_prefix is not None:
        best_res = load_or_save_dict(f"{save_prefix}_{save_midfix}_best.pkl", _main, n_clus_lim=n_clus_lim)
    else:
        best_res = _main(n_clus_lim)

    return best_res


# clustering

def clustering(data_mat, clus_algo,
               order_label=False,  # whether to order label
               **clus_kwargs):
    if clus_algo == "kmeans":
        clus_res = kmeans_sort(data_mat, **clus_kwargs)
    else:
        assert False

    n_clus = clus_res['n_clus'];
    clus_labels = clus_res['res']
    if n_clus > 1:
        sscores = silhouette_samples(data_mat, clus_labels, metric=clus_kwargs['dis_metric'])
    else:
        sscores = np.zeros(len(data_mat))
    if order_label:
        # order the label by the mean silhouette score
        sscores_mean = [np.mean(sscores[clus_labels == li]) for li in range(n_clus)]
        labels_sorted = np.argsort(sscores_mean)[::-1]
        labels_sorted = {lidx: li for li, lidx, in enumerate(labels_sorted)}
        clus_labels = np.array([labels_sorted[lidx] for lidx in clus_labels])
        clus_res['res'] = clus_labels  # rewrite
    # order the samples first by the clus label, then by the silhouette score
    y_sort = np.concatenate(
        [np.where(clus_labels == li)[0][np.argsort(sscores[clus_labels == li])[::-1]] for li in range(n_clus)])
    clus_res['sscores_sample'] = sscores
    clus_res['y_sort'] = y_sort

    return clus_res


"""
input:
    - coef_vs: [Nneuron, Nvariables]
    - session id (eid) of the corresponding neuron: [Nneuron], used in detecting the suspecious cluster
    - clus_param: parameter used for the clustering algorithm
        - for kmeans: 
            n_clus_lim, a list of 2 items specifying the minimum and maximum number of clusters
            n_init, int 
    - clus_folder: path to the folder for saving the clustering results
    - area: name of the area, used to name the file for saving the clustering results
    - remove_susclus: whether to remove the suspecious cluster
    - susclus_thres: proportion of neurons belonging to one session for the cluster to be removed
    - minN: stop clustering analysis if the number of neurons is smaller than this value
return:
    - dict:
        - success: whether the clustering analysis is performed successfully or not
        - data_clus_hist: list of clustering history (as we remove the suspecious cluster iteratively), each term contains the following values:
            - coef_vs: [Nneuron, Nvariable] data matrix used in this round of clustering
            - res: [Nneuron] corresponding cluster labels
            - n_clus: int, number of clusters
            - sscores_sample: [Nneuron] silhouette score of each neuron
            - ...
"""


def cluster_analysis_old(coef_vs, eids, clus_param,
                     clus_folder, area,
                     remove_susclus=True, susclus_thres=0.9,
                     minN=25):
    ### iterate until no suspecious cluster of single session is found
    roundi = 0;
    data_clus_hist = []
    while True:
        # we only perform the clustering analysis if there are more than >= minN number of neurons
        if len(coef_vs) < minN:
            return dict(data_clus_hist=data_clus_hist, success=False)

        clus_res = clustering(coef_vs,
                              save_prefix=os.path.join(clus_folder, f"{area}_round{roundi}"),
                              order_label=True,
                              **clus_param)
        hist = dict(coef_vs=coef_vs, **clus_res)

        # the session problem
        suspicious_clus = []
        for _li in np.unique(clus_res['res']):
            _li_mask = clus_res['res'] == _li
            for _eid in np.unique(eids[_li_mask]):
                _mask = (clus_res['res'] == _li) & (eids == _eid)
                if np.sum(clus_res['sscores_sample'][_mask]) / np.sum(
                        clus_res['sscores_sample'][_li_mask]) >= susclus_thres:
                    print(f"{area} label {_li} is suspicious!")
                    suspicious_clus.append(_li)
        suspicious_nismask = np.isin(clus_res['res'], suspicious_clus)
        hist['eids'] = eids
        hist['suspicious_clus'] = suspicious_clus
        hist['suspicious_nismask'] = suspicious_nismask

        data_clus_hist.append(hist)

        if (remove_susclus) and len(suspicious_clus) > 0:
            # rewrite the coef_vs 
            # and
            # reclustering
            coef_vs = coef_vs[~suspicious_nismask]
            eids = eids[~suspicious_nismask]
            roundi += 1
        else:
            # success, return
            return dict(data_clus_hist=data_clus_hist, success=True)


"""
input: 
    check the function: cluster_analysis
return:
    - sscores_nulls: list of silhouette score of Nnull null models
"""


def cluster_null(coef_vs, clus_param, Nnull,
                 clus_folder, area):
    sscores_nulls = []
    for i in tqdm(range(Nnull)):
        np.random.seed(42 + i)
        coef_vs_null = np.random.multivariate_normal(np.mean(coef_vs, 0), np.cov(coef_vs.T), len(coef_vs))
        clus_res = clustering(coef_vs_null,
                              save_prefix=os.path.join(clus_folder, f"nullG_{area}_{i}"),
                              order_label=True,
                              **clus_param)
        sscores_nulls.append(np.mean(clus_res['sscores_sample']))
    return sscores_nulls


"""
input:
    same as in cluster_analysis.
return:
    sscore_z: float
"""


def sscore_z(coef_vs, eids, clus_param, Nnull,
             clus_folder, area,
             remove_susclus=True, susclus_thres=0.9,
             minN=25):
    hist = cluster_analysis(coef_vs, eids, clus_param,
                            clus_folder, area,
                            remove_susclus=remove_susclus, susclus_thres=susclus_thres,
                            minN=minN)

    if hist['success'] == False:
        return 0.
    else:
        sscores_nulls = cluster_null(hist['data_clus_hist'][-1]['coef_vs'], clus_param, Nnull,
                                     clus_folder, area)
        sscore_z = (np.mean(hist['data_clus_hist'][-1]['sscores_sample']) - np.mean(sscores_nulls)) / np.std(
            sscores_nulls)
        return sscore_z


# Random clustered data generator

def generate_clustered_data(num_trials, num_matrices, matrix_shapes, cluster_centers, cluster_spreads):
    conditioned_trials = []

    for _ in range(num_trials):
        trial = {}
        for i in range(num_matrices):
            rows, cols = matrix_shapes[i]
            cluster_center = cluster_centers[i]
            cluster_spread = cluster_spreads[i]

            # Create a base matrix with random values
            matrix = np.random.randn(rows, cols) * cluster_spread

            # Add clustering effect by increasing values in certain columns
            for col in range(cols):
                matrix[:, col] += cluster_center[col]

            trial[f'matrix{i + 1}'] = matrix

        conditioned_trials.append(trial)

    return conditioned_trials


def collapse_raster_to_target(raster, target_tr_mean, collapse_alpha):
    tr_mean = np.nanmean(raster, 0)
    delta_tr = tr_mean - target_tr_mean
    collapsed_raster = raster - collapse_alpha * delta_tr
    return collapsed_raster


def collapse_trials(conditioned_trials, cluster_labels, alpha):
    target_means = {}
    for key in conditioned_trials[0]:
        x = np.hstack([np.nanmean(c[key], 0) for c in conditioned_trials])
        xc = np.zeros(len(x))
        for c in np.unique(cluster_labels):
            cmask = cluster_labels == c
            xc[cmask] = np.nanmean(x[cmask])
        target_means[key] = xc

    idx = 0

    collapsed_conditioned_trials = []
    for c in conditioned_trials:
        t = c[list(c.keys())[0]].shape[1]
        collapsed_c = {}
        for key in c:
            collapsed_c[key] = collapse_raster_to_target(c[key], target_means[key][idx:idx+t], alpha)
        collapsed_conditioned_trials.append(collapsed_c)
        idx += t

    return collapsed_conditioned_trials
