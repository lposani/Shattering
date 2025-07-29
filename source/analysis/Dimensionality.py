import os
import pickle

import numpy as np
import pandas as pd
from decodanda.utilities import *
from source.utilities import *
from sklearn.decomposition import PCA


def shuffle_conditioned_trials(conditioned_trials):
    shuffled_conditioned_trials = []
    for c in conditioned_trials:
        shuffled_c = {}
        all_conditions = list(c.keys())
        all_data = np.vstack([c[cond] for cond in all_conditions])
        all_n_trials = {cond: len(c[cond]) for cond in all_conditions}
        trials = np.arange(all_data.shape[0])
        np.random.shuffle(trials)
        all_data = all_data[trials]

        i = 0
        for cond in all_conditions:
            shuffled_c[cond] = all_data[i:i + all_n_trials[cond]]
            i += all_n_trials[cond]
        shuffled_conditioned_trials.append(shuffled_c)
    return shuffled_conditioned_trials


def conditioned_trial_to_X(conditioned_trials, min_n_trials, conditions_subset=None):
    X = {}
    if conditions_subset is None:
        conditions_subset = conditioned_trials.keys()
    for k in conditioned_trials:
        if k in conditions_subset:
            if len(conditioned_trials[k]) > min_n_trials:
                # print(k, len(conditioned_trials[k]))
                randidx = np.arange(len(conditioned_trials[k]))
                np.random.shuffle(randidx)
                X[k] = conditioned_trials[k][randidx[:min_n_trials], :]
    return X


def visualize_centroids_PCA(conditioned_trials, min_n_trials, color_lambda=None, alpha_lambda=None):
    labels = []
    X = []
    for k in conditioned_trials:
        print(''.join(k), len(conditioned_trials[k]))
        if len(conditioned_trials[k]) > min_n_trials:
            labels.append(''.join(k))
            X.append(np.nanmean(conditioned_trials[k], 0))
    X = np.vstack(X)
    labels = np.hstack(labels)
    X = z_score_raster(X)
    C = PCA(n_components=3)
    X3 = C.fit_transform(X)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection='3d')

    for i in range(len(labels)):
        l = labels[i]
        if color_lambda is not None:
            if color_lambda(l):
                if alpha_lambda is not None:
                    if alpha_lambda(l):
                        ax.scatter(X3[i, 0], X3[i, 1], X3[i, 2], marker=f'${l}$', s=1500, color=pltcolors[0],
                                   alpha=0.35)
                    else:
                        ax.scatter(X3[i, 0], X3[i, 1], X3[i, 2], marker=f'${l}$', s=1500, color=pltcolors[0])
                else:
                    ax.scatter(X3[i, 0], X3[i, 1], X3[i, 2], marker=f'${l}$', s=1500, color=pltcolors[0])
            else:
                if alpha_lambda is not None:
                    if alpha_lambda(l):
                        ax.scatter(X3[i, 0], X3[i, 1], X3[i, 2], marker=f'${l}$', s=1500, color=pltcolors[3],
                                   alpha=0.35)
                    else:
                        ax.scatter(X3[i, 0], X3[i, 1], X3[i, 2], marker=f'${l}$', s=1500, color=pltcolors[3])
                else:
                    ax.scatter(X3[i, 0], X3[i, 1], X3[i, 2], marker=f'${l}$', s=1500, color=pltcolors[3])
        else:
            ax.scatter(X3[i, 0], X3[i, 1], X3[i, 2], marker=f'${l}$', s=1500, color='k')


def collapse_raster(raster, cluster_labels, collapse_alpha):
    tr_mean = np.nanmean(raster, 0)
    target_tr_mean = np.zeros(raster.shape[1])
    for c in np.unique(cluster_labels):
        cmask = cluster_labels == c
        target_tr_mean[cmask] = np.nanmean(tr_mean[cmask])
    delta_tr = tr_mean - target_tr_mean
    collapsed_raster = raster - collapse_alpha * delta_tr
    return collapsed_raster


def collapse_raster_to_target(raster, target_tr_mean, collapse_alpha):
    tr_mean = np.nanmean(raster, 0)
    delta_tr = tr_mean - target_tr_mean
    collapsed_raster = raster - collapse_alpha * delta_tr
    return collapsed_raster


def collapse_raster_subspace(raster, subspace_vectors):
    """
    Moves the centroid of the raster onto the given subspace,
    but preserves all relative variability (trial-to-trial structure).

    Parameters:
    - raster: array of shape (n_trials, n_neurons)
    - subspace_vectors: array of shape (k, n_neurons), assumed orthonormal

    Returns:
    - collapsed_raster: array of shape (n_trials, n_neurons)
    """

    # Compute the centroid of the data
    mean_vector = np.nanmean(raster, axis=0)

    # Project the centroid onto the subspace
    coeffs = mean_vector @ subspace_vectors.T  # shape: [1, k]
    projected_mean = coeffs @ subspace_vectors  # shape: [1, n_neurons]

    # Keep original deviations from the mean
    deviations = raster - mean_vector

    # Final result: same deviations, new centroid
    collapsed_raster = projected_mean + deviations
    return collapsed_raster


def semantic_score2(dichotomy, convert_IBL=True):
    # convert the dichotomy from string to binary
    if convert_IBL:
        mappings = [
            {'S': 0, 'W': 1},
            {'2': 0, '8': 1},
            {'<': 0, '>': 1},
            {'h': 1, 'l': 0}
        ]

        dic = []
        for sublist in dichotomy:
            binary_sublist = []
            for tuple_ in sublist:
                binary_tuple = [mappings[i][value] for i, value in enumerate(tuple_)]
                binary_sublist.append(binary_tuple)
            dic.append(binary_sublist)
            d = dic[0]
    else:
        d = [string_bool(x) for x in dichotomy[0]]

    # convert the dichotomy from string to binary
    fingerprint = np.abs(np.sum(d, 0) - len(d) / 2)
    return fingerprint  # (np.max(fingerprint) - np.mean(fingerprint))/((len(d) / 2)*(1-1./(len(fingerprint))))


def decode_dichotomy(conditioned_trials, dichotomy=None,
                     n_neurons=None, zscore=False, classifier=None,
                     subspace=None, shuffled=False, **decoding_params):

    # If unspecified: use a random dichotomy
    keys = list(conditioned_trials[0].keys())
    if dichotomy is None:
        randkeys = list(conditioned_trials[0].keys())
        np.random.shuffle(randkeys)
        dichotomy = [randkeys[:int(len(randkeys) / 2)], randkeys[int(len(randkeys) / 2):]]

    # Implement a balanced decoding similar to the one in Decodanda
    conditioned_rasters = {key: [c[key] for c in conditioned_trials] for key in keys}

    # Assume all trials are independent
    conditioned_trial_index = {key: [np.arange(len(c[key])) for c in conditioned_trials] for key in keys}

    set_A = dichotomy[0]
    set_B = dichotomy[1]

    n_conditions_A = float(len(set_A))
    n_conditions_B = float(len(set_B))
    fraction = n_conditions_A / n_conditions_B

    # Begin cross-validations
    perfs = []
    for _ in range(decoding_params['cross_validations']):
        viz_labels = []
        training_trials = {}  # for subspace projection
        testing_trials = {}
        training_array_A = []
        training_array_B = []
        testing_array_A = []
        testing_array_B = []
        # Balanced sample from the first side of the dichotomy
        for d in set_A:
            training, testing = sample_training_testing_from_rasters(conditioned_rasters[d],
                                                                     int(decoding_params['ndata'] / fraction),
                                                                     decoding_params['training_fraction'],
                                                                     conditioned_trial_index[d])

            if 'break_correlations' in decoding_params and decoding_params['break_correlations']:
                for n in range(training.shape[1]):
                    idx = np.arange(training.shape[0])
                    np.random.shuffle(idx)
                    training[:, n] = training[idx, n]
                for n in range(testing.shape[1]):
                    idx = np.arange(testing.shape[0])
                    np.random.shuffle(idx)
                    testing[:, n] = testing[idx, n]

            for k in range(int(decoding_params['ndata'] / fraction)):
                viz_labels.append(''.join(d))

            if 'cluster_labels' in decoding_params:
                training = collapse_raster(training, decoding_params['cluster_labels'],
                                           decoding_params['collapse_alpha'])
                testing = collapse_raster(testing, decoding_params['cluster_labels'],
                                          decoding_params['collapse_alpha'])

            training_trials[d] = training
            testing_trials[d] = testing

            training_array_A.append(training)
            testing_array_A.append(testing)

        # Balanced sample from the second side of the dichotomy
        for d in set_B:
            training, testing = sample_training_testing_from_rasters(conditioned_rasters[d],
                                                                     int(decoding_params['ndata']),
                                                                     decoding_params['training_fraction'],
                                                                     conditioned_trial_index[d])

            if 'break_correlations' in decoding_params and decoding_params['break_correlations']:
                for n in range(training.shape[1]):
                    idx = np.arange(training.shape[0])
                    np.random.shuffle(idx)
                    training[:, n] = training[idx, n]
                for n in range(testing.shape[1]):
                    idx = np.arange(testing.shape[0])
                    np.random.shuffle(idx)
                    testing[:, n] = testing[idx, n]

            for k in range(int(decoding_params['ndata'])):
                viz_labels.append(''.join(d))

            if 'cluster_labels' in decoding_params and decoding_params['collapse_alpha']:
                training = collapse_raster(training, decoding_params['cluster_labels'],
                                           decoding_params['collapse_alpha'])
                testing = collapse_raster(testing, decoding_params['cluster_labels'],
                                          decoding_params['collapse_alpha'])

            training_trials[d] = training
            testing_trials[d] = testing

            training_array_B.append(training)
            testing_array_B.append(testing)

        # Subsample to a given number of neurons
        if n_neurons is not None:
            subset = np.random.choice(training_array_A[0].shape[1], n_neurons, replace=False)
            for k in training_trials:
                training_trials[k] = training_trials[k][:, subset]
                testing_trials[k] = testing_trials[k][:, subset]
            training_array_A = [t[:, subset] for t in training_array_A]
            training_array_B = [t[:, subset] for t in training_array_B]
            testing_array_A = [t[:, subset] for t in testing_array_A]
            testing_array_B = [t[:, subset] for t in testing_array_B]

        if subspace is not None:
            from source.analysis.Clustering import CT_to_X

            # Collapse training data
            C = sklearn.decomposition.PCA(n_components=subspace)
            X = CT_to_X(training_trials, zscore=False)
            C.fit(X)

            collapsed_training_array_A = []
            for t in training_array_A:
                collapsed_training_array_A.append(C.transform(t))
            training_array_A = np.vstack(collapsed_training_array_A)

            collapsed_training_array_B = []
            for t in training_array_B:
                collapsed_training_array_B.append(C.transform(t))
            training_array_B = np.vstack(collapsed_training_array_B)


            # Collapse testing data
            collapsed_testing_array_A = []
            for t in testing_array_A:
                collapsed_testing_array_A.append(C.transform(t))
            testing_array_A = np.vstack(collapsed_testing_array_A)

            collapsed_testing_array_B = []
            for t in testing_array_B:
                collapsed_testing_array_B.append(C.transform(t))
            testing_array_B = np.vstack(collapsed_testing_array_B)

        else:
            training_array_A = np.vstack(training_array_A)
            training_array_B = np.vstack(training_array_B)
            testing_array_A = np.vstack(testing_array_A)
            testing_array_B = np.vstack(testing_array_B)

        # visualize_representations_PCA(np.vstack([training_array_A, training_array_B]), dim=3, labels=np.asarray(viz_labels), show_centroids=True)

        if zscore:
            big_raster = np.vstack([training_array_A, training_array_B])  # z-scoring using the training data
            big_mean = np.nanmean(big_raster, 0)
            big_std = np.nanstd(big_raster, 0)
            big_std[big_std == 0] = np.inf
            training_array_A = (training_array_A - big_mean) / big_std
            training_array_B = (training_array_B - big_mean) / big_std
            testing_array_A = (testing_array_A - big_mean) / big_std
            testing_array_B = (testing_array_B - big_mean) / big_std

        label_A = '1'
        label_B = '0'

        # Train the classifier
        if classifier is None:
            classifier = LinearSVC(dual=False, C=1.0, class_weight='balanced', max_iter=5000)

        training_labels_A = np.repeat(label_A, training_array_A.shape[0]).astype(object)
        training_labels_B = np.repeat(label_B, training_array_B.shape[0]).astype(object)
        training_raster = np.vstack([training_array_A, training_array_B])
        training_labels = np.hstack([training_labels_A, training_labels_B])

        testing_labels_A = np.repeat(label_A, testing_array_A.shape[0]).astype(object)
        testing_labels_B = np.repeat(label_B, testing_array_B.shape[0]).astype(object)
        testing_raster = np.vstack([testing_array_A, testing_array_B])
        testing_labels = np.hstack([testing_labels_A, testing_labels_B])

        if shuffled:
            np.random.shuffle(training_labels)
            np.random.shuffle(testing_labels)

        classifier.fit(training_raster, training_labels)
        performance = classifier.score(testing_raster, testing_labels)
        perfs.append(performance)

    return np.asarray(perfs)


def sample_PS(conditioned_trials, ndata, z_score=True, cs=None):
    if type(conditioned_trials) is not list:
        conditioned_trials = [conditioned_trials]

    conditions = list(conditioned_trials[0].keys())

    # convert to key: list instead of list, key
    conditioned_rasters = {k: [c[k] for c in conditioned_trials] for k in conditions}

    rasters = []
    labels = []
    if cs is None:
        cs = [''.join(key) for key in conditions]

    for i, key in enumerate(conditions):
        if ''.join(key) in cs:
            X = sample_from_rasters(conditioned_rasters[key], ndata)
            rasters.append(X)
            labels.append(np.repeat(''.join(key), ndata))
    X = np.vstack(rasters)
    y = np.hstack(labels)

    if z_score:
        for i in range(X.shape[1]):
            if np.nanstd(X[:, i]):
                X[:, i] = (X[:, i] - np.nanmean(X[:, i])) / np.nanstd(X[:, i])

    return X, y


def participation_ratio(X):
    C = sklearn.decomposition.PCA()
    Xred = C.fit_transform(X)
    # XredT = C.fit_transform(X.T)
    sv = C.explained_variance_ratio_
    pratio = (np.sum(sv) ** 2) / np.sum(sv ** 2)
    return pratio


def semantic_dichotomies(keys):
    semantic_dics = []
    for v in range(len(keys[0])):
        dic = [[], []]
        for i in range(len(keys)):
            if floor(i / 2 ** v) % 2:
                dic[0].append(keys[i])
            else:
                dic[1].append(keys[i])
        semantic_dics.append(dic)
    return semantic_dics


def XORs(keys):
    sds = []
    P = len(keys)
    L = len(keys[0])
    for v in range(L):
        d = np.ones(P)
        for i in range(P):
            d[i] = (floor(i / 2 ** v) % 2) * 2 -1
        sds.append(d)
    sds = np.asarray(sds)

    half = P // 2
    xor_list = []

    for pos in combinations(range(P), half):
        x = -np.ones(P, dtype=int)
        x[list(pos)] = 1

        # Check orthogonality
        if np.all(sds @ x == 0):
            xor_list.append(x)
    return xor_list


def shattering_dimensionality(conditioned_trials, nreps, nnulls=100, n_neurons=None, region=None, folder=None,
                              IC=False, ax=None, convert_dic=True, subspace=None, add_semantic=False,
                              add_xors=False, cache_name=None, **decoding_params):
    if type(conditioned_trials) != list:
        conditioned_trials = [conditioned_trials]

    cache_name = f'./datasets/IBL/SD_cache/{cache_name}_N={n_neurons}'
    perfs = []
    perfs_null = []
    fingerprints = []
    keys = list(conditioned_trials[0].keys())

    # if type(subspace) == int:
    #     from source.analysis.Clustering import CT_to_X
    #     X = CT_to_X(conditioned_trials, zscore=True)
    #     C = sklearn.decomposition.PCA(n_components=subspace)
    #     C.fit(X)
    #     subspace = C.components_

    # Semantic Dichotomies
    if add_semantic:
        semantic_dics = semantic_dichotomies(keys)
        for i, dic in enumerate(semantic_dics):
            ipath = cache_name + f'_sem{i}.pck'
            if os.path.exists(ipath):
                [perf, fingerprint] = pickle.load(open(ipath, 'rb'))
            else:
                classifier = LinearSVC(dual=False, C=1.0, class_weight='balanced', max_iter=1000)
                perf = decode_dichotomy(conditioned_trials, dichotomy=dic,
                                        n_neurons=n_neurons, classifier=classifier,
                                        subspace=subspace,
                                        **decoding_params)

                fingerprint = semantic_score2(dic, convert_IBL=convert_dic)
                pickle.dump([perf, fingerprint], open(ipath, 'wb'))

            fingerprints.append(fingerprint)
            perfs.append(np.nanmean(perf))

    # Non-semantic Dichotomies
    if add_xors:
        print("Adding XORs")
        keys = list(conditioned_trials[0].keys())
        xor_dics = XORs(keys)
        # xor_dics = xor_dics[:100]

        for i, dic in tqdm(enumerate(xor_dics)):
            ipath = cache_name + f'_xor{i}.pck'
            if os.path.exists(ipath):
                [perf, fingerprint] = pickle.load(open(ipath, 'rb'))
            else:
                classifier = LinearSVC(dual=False, C=1.0, class_weight='balanced', max_iter=1000)
                perf = decode_dichotomy(conditioned_trials, dichotomy=dic,
                                        n_neurons=n_neurons, classifier=classifier,
                                        subspace=subspace,
                                        **decoding_params)
                print(perf)
                fingerprint = semantic_score2(dic, convert_IBL=convert_dic)
                pickle.dump([perf, fingerprint], open(ipath, 'wb'))

            fingerprints.append(fingerprint)
            perfs.append(np.nanmean(perf))

    # Random Dichotomies
    if nreps is not None:
        allpath = cache_name + f'_data_randdics_IC={IC}_nreps={nreps}.pck'
        if os.path.exists(allpath):
            [perfs, fingerprints] = pickle.load(open(allpath, 'rb'))
            print('loading', allpath)
        else:
            print(allpath, 'not found, computing dichotomies')
            for i in tqdm(range(nreps)):
                ipath = cache_name + f'_data_randdics_IC={IC}_{i}.pck'
                if os.path.exists(ipath):
                    [perf, dichotomy] = pickle.load(open(ipath, 'rb'))
                else:
                    classifier = LinearSVC(dual=False, C=1.0, class_weight='balanced', max_iter=1000)

                    # Use a random dichotomy
                    randkeys = list(conditioned_trials[0].keys())
                    np.random.shuffle(randkeys)
                    dichotomy = [randkeys[:int(len(randkeys) / 2)], randkeys[int(len(randkeys) / 2):]]

                    perf = decode_dichotomy(conditioned_trials, dichotomy=dichotomy,
                                            n_neurons=n_neurons, classifier=classifier,
                                            subspace=subspace,
                                            **decoding_params)
                    pickle.dump([perf, dichotomy], open(ipath, 'wb'))

                perfs.append(np.nanmean(perf))
                if not IC:
                    fingerprint = semantic_score2(dichotomy, convert_IBL=convert_dic)
                    fingerprints.append(fingerprint)
                    print(np.nanmean(perf), fingerprint)

            pickle.dump([perfs, fingerprints], open(allpath, 'wb'))
            for i in range(nreps):
                ipath = cache_name + f'_data_randdics_IC={IC}_{i}.pck'
                os.remove(ipath)
    else:
        dichotomies = balanced_dichotomies(keys)
        allpath = cache_name + f'_data_alldics_IC={IC}_all.pck'
        if os.path.exists(allpath):
            [perfs, fingerprints] = pickle.load(open(allpath, 'rb'))
            print('loading', allpath)
        else:
            for i in tqdm(range(len(dichotomies))):
                ipath = cache_name + f'_data_alldics_IC={IC}_{i}.pck'
                if os.path.exists(ipath):
                    [perf, dichotomy] = pickle.load(open(ipath, 'rb'))
                else:
                    classifier = LinearSVC(dual=False, C=1.0, class_weight='balanced', max_iter=1000)

                    dichotomy = dichotomies[i]
                    perf = decode_dichotomy(conditioned_trials, dichotomy=dichotomy, n_neurons=n_neurons,
                                            classifier=classifier,
                                            subspace=subspace,
                                            **decoding_params)
                    pickle.dump([perf, dichotomy], open(ipath, 'wb'))

                perfs.append(np.nanmean(perf))
                if not IC:
                    fingerprint = semantic_score2(dichotomy, convert_IBL=convert_dic)
                    fingerprints.append(fingerprint)

            # Save aggregated data and remove individual files
            pickle.dump([perfs, fingerprints], open(allpath, 'wb'))
            for i in range(len(dichotomies)):
                ipath = cache_name + f'_data_alldics_IC={IC}_{i}.pck'
                os.remove(ipath)

    # Null Model
    allnullpath = cache_name + f'_null_IC={IC}_{nnulls}.pck'
    if os.path.exists(allnullpath):
        perfs_null = pickle.load(open(allnullpath, 'rb'))
        print('loading', allnullpath)
    else:
        for i in tqdm(range(nnulls)):
            ipath = cache_name + f'_null_IC={IC}_{i}.pck'
            if os.path.exists(ipath):
                [perf_null, dichotomy] = pickle.load(open(ipath, 'rb'))
            else:
                classifier = LinearSVC(dual=False, C=1.0, class_weight='balanced', max_iter=1000)
                # If unspecified: use a random dichotomy
                randkeys = list(conditioned_trials[0].keys())
                np.random.shuffle(randkeys)
                dichotomy = [randkeys[:int(len(randkeys) / 2)], randkeys[int(len(randkeys) / 2):]]

                shuffled_conditioned_trials = shuffle_conditioned_trials(conditioned_trials)
                perf_null = decode_dichotomy(shuffled_conditioned_trials,
                                             dichotomy=dichotomy, n_neurons=n_neurons,
                                             classifier=classifier, subspace=subspace,
                                             **decoding_params)
                pickle.dump([perf_null, dichotomy], open(ipath, 'wb'))

            perfs_null.append(np.nanmean(perf_null))

        # Save aggregated and delete
        pickle.dump(perfs_null, open(allnullpath, 'wb'))
        for i in tqdm(range(nnulls)):
            ipath = cache_name + f'_null_IC={IC}_{i}.pck'
            os.remove(ipath)

    ntop = np.percentile(perfs_null, 99)
    nbot = np.percentile(perfs_null, 1)

    if ax is not None:
        ax.hist(perfs, bins=np.linspace(0.4, 1.0, 25), alpha=0.5, label=region, density=True)
        ax.set_xlabel('Decoding Performance')
        ax.set_ylabel('Count')
        ax.set_xlim([0.4, 1.0])
        ys5 = ax.get_ylim()
        ax.fill_between(x=[nbot, ntop], color='k', alpha=0.1,
                        y1=ys5[0], y2=ys5[1])
        ax.axvline([nbot], color='k', linestyle='--', alpha=0.5)
        ax.axvline([ntop], color='k', linestyle='--', alpha=0.5)
        ax.text(1, ys5[1], 'SD=%.2f' % np.nanmean(perfs > ntop), ha='right')
        ax.set_ylim(ys5)
        linenull(ax=ax, direction='v', null=np.nanmean(perfs_null))

    if region is not None and folder is not None:
        if not IC:
            fingerprints = np.asarray(fingerprints)
            semantic_scores = np.sqrt(np.sum(fingerprints ** 2, 1))
            L = len(keys[0])
            labels = []
            for ss in semantic_scores:
                if ss == 0:
                    labels.append('XOR')
                elif ss == 2**(L-2):
                    labels.append('Sem')
                else:
                    labels.append('Mix')

            f = plt.figure(figsize=(12, 5))
            g = GridSpec(21, 30, figure=f)
            axs = [
                f.add_subplot(g[:12, 1:5]),
                f.add_subplot(g[:8, 9:13]),
                f.add_subplot(g[:8, 15:19]),
                f.add_subplot(g[12:-1, 9:13]),
                f.add_subplot(g[12:-1, 15:19]),
                f.add_subplot(g[14:, 1:6]),
                f.add_subplot(g[:8, 22:]),
                f.add_subplot(g[12:, 24:-2]),
            ]
            sns.despine(f)

            axs[5].hist(perfs, bins=np.linspace(0.4, 1.0, 25), alpha=0.5, label=region, density=True)
            axs[5].set_xlabel('Decoding Performance')
            axs[5].set_ylabel('Count')

            axs[5].axvline([nbot], color='k', linestyle='--', alpha=0.5)
            axs[5].axvline([ntop], color='k', linestyle='--', alpha=0.5)
            ys5 = axs[5].get_ylim()
            axs[5].fill_between(x=[nbot, ntop], color='k', alpha=0.1,
                                y1=ys5[0], y2=ys5[1])
            axs[5].set_ylim(ys5)
            axs[5].text(1, ys5[1], 'SD=%.2f' % np.nanmean(perfs > ntop), ha='right')
            linenull(ax=axs[5], direction='v', null=np.nanmean(perfs_null))

            sns.swarmplot(data=pd.DataFrame.from_dict({'Performance': perfs, 'Class': labels}),
                          y='Performance', ax=axs[0], hue='Class',
                          palette={'Mix': [0.8, 0.6, 0.6, 0.6], 'XOR': pltcolors[2], 'Sem': 'magenta'})

            axs[0].legend(fontsize=7, loc='lower right', bbox_to_anchor=(1, 1))
            axs[0].set_ylabel('Decoding Performance')
            axs[0].set_ylim([0.4, 1.0])
            axs[0].set_xticklabels([region])
            axs[0].fill_between(x=[-1, 1], y1=nbot, y2=ntop, color='k',
                                alpha=0.1)
            axs[0].axhline([nbot], color='k', linestyle='--', alpha=0.5)
            axs[0].axhline([ntop], color='k', linestyle='--', alpha=0.5)
            linenull(axs[0], np.nanmean(perfs_null))
            axs[0].set_xlim([-0.9, 0.9])

            if convert_dic:
                corr_scatter(fingerprints[:, 0], perfs, 'Semanticity: Whisking', 'Decoding Performance', ax=axs[1],
                             alpha=0.5, marker='.')
                corr_scatter(fingerprints[:, 1], perfs, 'Semanticity: Block', '', ax=axs[2], alpha=0.5, marker='.')
                corr_scatter(fingerprints[:, 2], perfs, 'Semanticity: Side', 'Decoding Performance', ax=axs[3], alpha=0.5,
                             marker='.')
                corr_scatter(fingerprints[:, 3], perfs, 'Semanticity: Contrast', '', ax=axs[4], alpha=0.5, marker='.')
            else:
                if len(keys[0]) <= 4:
                    for k in range(len(keys[0])):
                        corr_scatter(fingerprints[:, k], perfs, f'Semanticity: Var {k}', 'Decoding Performance',
                                     ax=axs[1+k],
                                     alpha=0.5, marker='.')

            sns.swarmplot(data=pd.DataFrame.from_dict({'Performance': perfs, 'Semantic Class': labels}),
                          x='Semantic Class', y='Performance', ax=axs[6],
                          palette={'XOR': pltcolors[2], 'Mix': [0.8, 0.6, 0.6, 0.6], 'Sem': 'magenta'},
                          order=['XOR', 'Mix', 'Sem'])
            sns.pointplot(data=pd.DataFrame.from_dict({'Performance': perfs, 'Semantic Class': labels}),
                          x='Semantic Class', y='Performance', ax=axs[6],
                          palette={'XOR': pltcolors[2], 'Mix': [0.8, 0.6, 0.6, 0.6], 'Sem': 'magenta'},
                          linestyle='', alpha=0.5, capsize=.3, marker='_', order=['XOR', 'Mix', 'Sem'])

            axs[6].fill_between(x=[-1, 4], y1=nbot, y2=ntop, color='k',
                                alpha=0.1)
            axs[6].axhline([nbot], color='k', linestyle='--', alpha=0.5)
            axs[6].axhline([ntop], color='k', linestyle='--', alpha=0.5)
            linenull(axs[6], np.nanmean(perfs_null))
            axs[6].set_xlim([-0.5, 2.5])
            axs[6].set_ylim([0.4, 1.0])

            r2, p = corr_scatter(semantic_scores, perfs, 'Semantic Score', 'Decoding Performance', alpha=0.3, ax=axs[7])

            plt.suptitle(f'Region: {region}')
            plt.savefig(f'./plots/IBL/{folder}/SD/perfs_{region}.pdf')
            plt.close(f)

        else:
            f, axs = plt.subplots(1, 2, figsize=(5, 3), gridspec_kw={'width_ratios': [1, 2]})
            sns.despine(f)

            sns.swarmplot(perfs, ax=axs[0])
            axs[0].set_ylabel('Decoding Performance')
            axs[0].set_ylim([0.4, 1.0])
            axs[0].set_xticklabels([region])
            axs[0].fill_between(x=[-1, 1], y1=nbot, y2=ntop, color='k',
                                alpha=0.1)
            axs[0].axhline([nbot], color='k', linestyle='--', alpha=0.5)
            axs[0].axhline([ntop], color='k', linestyle='--', alpha=0.5)
            linenull(axs[0], np.nanmean(perfs_null))
            axs[0].set_xlim([-0.9, 0.9])

            ax = axs[1]
            ax.hist(perfs, bins=np.linspace(0.4, 1.0, 25), alpha=0.5, label=region, density=True)
            ax.set_xlabel('Decoding Performance')
            ax.set_ylabel('Count')
            ax.set_xlim([0.4, 1.0])
            ys5 = axs[1].get_ylim()
            axs[1].fill_between(x=[nbot, ntop], color='k', alpha=0.1,
                                y1=ys5[0], y2=ys5[1])
            axs[1].axvline([nbot], color='k', linestyle='--', alpha=0.5)
            axs[1].axvline([ntop], color='k', linestyle='--', alpha=0.5)
            axs[1].text(1, ys5[1], 'SD=%.2f\n$\left<DP\\right>=%.2f$' % (np.nanmean(perfs > ntop), np.nanmean(perfs)), ha='right')
            axs[1].set_ylim(ys5)
            linenull(ax=axs[1], direction='v', null=np.nanmean(perfs_null))
            axs[1].axvline(np.nanmean(perfs), color='r')
            plt.suptitle(f'Region: {region}')
            plt.savefig(f'./plots/IBL/{folder}/SD/perfs_{region}_IC.pdf')

            plt.close(f)

    return np.asarray(perfs), np.asarray(perfs_null), fingerprints


def decoding_matrix(conditioned_trials, min_data, labels=None, plot_mds=False, plot_matrix=True, precomputed_perfs=None,
                    **decoding_params):
    if precomputed_perfs is None:
        precomputed_perfs = {}

    conditioned_trials = {k: conditioned_trials[k] for k in conditioned_trials if len(conditioned_trials[k]) > min_data}
    if labels is None:
        labels = list(conditioned_trials.keys())
        labels_str = [''.join(k) for k in conditioned_trials]
    else:
        labels_str = [''.join(k) for k in labels]

    D = np.zeros((len(labels_str), len(labels_str)))
    Dz = np.zeros((len(labels_str), len(labels_str)))

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            xi = conditioned_trials[labels[i]]
            xj = conditioned_trials[labels[j]]
            pair_key = f'{labels[i]}-{labels[j]}'
            if pair_key not in precomputed_perfs:
                raster = np.vstack([xi, xj])
                y = np.hstack([np.zeros(len(xi)), np.ones(len(xj))])
                trial = np.arange(len(y))
                dec = Decodanda(data={'raster': raster, 'y': y, 'trial': trial},
                                conditions={'y': [0, 1]})
                perf, null = dec.decode_with_nullmodel(dichotomy='y', **decoding_params)
                precomputed_perfs[pair_key] = [perf, null]
            else:
                perf, null = precomputed_perfs[pair_key]

            z, p = z_pval(perf, null)
            print('%.2f' % perf, p_to_text(p))
            Dz[i, j] = z
            Dz[j, i] = z
            D[i, j] = perf
            D[j, i] = perf

    if plot_matrix:
        f, axs = plt.subplots(2, 2, figsize=(10, 8))
        ax = axs[0, 0]
        g = ax.pcolor(D, vmin=0.5, vmax=1.0)
        cb = plt.colorbar(g)
        cb.set_label('Decoding Performance')
        ax.set_xticks(np.arange(len(labels_str)) + 0.5)
        ax.set_yticks(np.arange(len(labels_str)) + 0.5)
        ax.set_xticklabels(labels_str, rotation=45, ha='right', va='top')
        ax.set_yticklabels(labels_str, rotation=0, ha='right', va='center')
        ax.set_title('Decoding performance')

        Ds, idx = sort_matrix(D)
        ax = axs[0, 1]
        g = ax.pcolor(Ds, vmin=0.5, vmax=1.0)
        cb = plt.colorbar(g)
        cb.set_label('Decoding Performance')
        ax.set_xticks(np.arange(len(labels_str)) + 0.5)
        ax.set_yticks(np.arange(len(labels_str)) + 0.5)
        ax.set_xticklabels(np.asarray(labels_str)[idx], rotation=45, ha='right', va='top')
        ax.set_yticklabels(np.asarray(labels_str)[idx], rotation=0, ha='right', va='center')
        ax.set_title('Decoding performance (sorted)')

        ax = axs[1, 0]
        g = ax.pcolor(Dz > 2.5)
        cb = plt.colorbar(g)
        cb.set_label('Decoding Performance (z-scored)')
        ax.set_xticks(np.arange(len(labels_str)) + 0.5)
        ax.set_yticks(np.arange(len(labels_str)) + 0.5)
        ax.set_xticklabels(labels_str, rotation=45, ha='right', va='top')
        ax.set_yticklabels(labels_str, rotation=0, ha='right', va='center')
        ax.set_title('Decoding performance (z-score)')

        Dzs, idx = sort_matrix(Dz)
        ax = axs[1, 1]
        g = ax.pcolor(Dzs > 2.5, vmin=0, vmax=1)
        cb = plt.colorbar(g)
        cb.set_label('Decoding Performance (z-scored)')
        ax.set_xticks(np.arange(len(labels_str)) + 0.5)
        ax.set_yticks(np.arange(len(labels_str)) + 0.5)
        ax.set_xticklabels(np.asarray(labels_str)[idx], rotation=45, ha='right', va='top')
        ax.set_yticklabels(np.asarray(labels_str)[idx], rotation=0, ha='right', va='center')
        ax.set_title('Decoding performance (z-score, sorted)')

        if plot_mds:
            # Perform Multi-Dimensional Scaling
            mds = MDS(n_components=3, dissimilarity='precomputed')
            positions = mds.fit_transform(D)

            # Plotting
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for i, l in enumerate(labels_str):
                if l[0] == '2' and l[3] == 'W':
                    color = 'red'
                elif l[0] == '2' and l[3] == 'S':
                    color = pltcolors[3]
                elif l[0] == '8' and l[3] == 'W':
                    color = 'blue'
                elif l[0] == '8' and l[3] == 'S':
                    color = pltcolors[0]
                else:
                    color = 'k'
                ax.scatter(positions[i, 0], positions[i, 1], positions[i, 2], marker='$' + l + '$', s=1200, c=color)

        return D, Dz, labels_str, f, axs

    return D, Dz, labels_str, precomputed_perfs


def decoding_matrix_PS(conditioned_trials, n_neurons=None, plot_mds=False,
                       plot_matrix=True, perf_threshold=None, precomputed_perfs=None,
                       **decoding_params):
    # assuming all conditioned trials in the list have the same keys!
    if type(conditioned_trials) is not list:
        conditioned_trials = [conditioned_trials]

    labels = list(conditioned_trials[0].keys())
    labels_str = [''.join(k) for k in conditioned_trials[0]]

    D = np.zeros((len(labels_str), len(labels_str)))
    Dz = np.zeros((len(labels_str), len(labels_str)))

    if precomputed_perfs is None:
        precomputed_perfs = {}

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            # create a list of sessions from conditioned trials
            pair_key = f'{labels[i]}-{labels[j]}'
            if pair_key not in precomputed_perfs:
                sessions = []
                for X in conditioned_trials:
                    xi = X[labels[i]]
                    xj = X[labels[j]]
                    raster = np.vstack([xi, xj])
                    y = np.hstack([np.zeros(len(xi)), np.ones(len(xj))])
                    trial = np.arange(len(y))
                    sessions.append({'raster': raster, 'y': y, 'trial': trial})
                dec = Decodanda(data=sessions,
                                conditions={'y': [0, 1]})
                if n_neurons is not None:
                    dec._generate_random_subset(n_neurons)
                perf, null = dec.decode_with_nullmodel(dichotomy='y', **decoding_params)
                precomputed_perfs[pair_key] = [perf, null]
            else:
                perf, null = precomputed_perfs[pair_key]
            D[i, j] = perf
            D[j, i] = perf

            if perf_threshold is None:
                z, p = z_pval(perf, null)
                # print('%.2f' % perf, p_to_text(p))
                Dz[i, j] = z
                Dz[j, i] = z
            else:
                Dz[i, j] = 3 * (perf > perf_threshold)
                Dz[j, i] = 3 * (perf > perf_threshold)
                # print('%.2f' % perf)

    if plot_matrix:
        f, axs = plt.subplots(2, 2, figsize=(10, 8))
        ax = axs[0, 0]
        g = ax.pcolor(D, vmin=0.5, vmax=1.0)
        cb = plt.colorbar(g)
        cb.set_label('Decoding Performance')
        ax.set_xticks(np.arange(len(labels_str)) + 0.5)
        ax.set_yticks(np.arange(len(labels_str)) + 0.5)
        ax.set_xticklabels(labels_str, rotation=45, ha='right', va='top')
        ax.set_yticklabels(labels_str, rotation=0, ha='right', va='center')
        ax.set_title('Decoding performance')

        Ds, idx = sort_matrix(D)
        ax = axs[0, 1]
        g = ax.pcolor(Ds, vmin=0.5, vmax=1.0)
        cb = plt.colorbar(g)
        cb.set_label('Decoding Performance')
        ax.set_xticks(np.arange(len(labels_str)) + 0.5)
        ax.set_yticks(np.arange(len(labels_str)) + 0.5)
        ax.set_xticklabels(np.asarray(labels_str)[idx], rotation=45, ha='right', va='top')
        ax.set_yticklabels(np.asarray(labels_str)[idx], rotation=0, ha='right', va='center')
        ax.set_title('Decoding performance (sorted)')

        ax = axs[1, 0]
        g = ax.pcolor(Dz > 2.5)
        cb = plt.colorbar(g)
        cb.set_label('Decoding Performance (z-scored)')
        ax.set_xticks(np.arange(len(labels_str)) + 0.5)
        ax.set_yticks(np.arange(len(labels_str)) + 0.5)
        ax.set_xticklabels(labels_str, rotation=45, ha='right', va='top')
        ax.set_yticklabels(labels_str, rotation=0, ha='right', va='center')
        ax.set_title('Decoding performance (z-score)')

        Dzs, idx = sort_matrix(Dz)
        ax = axs[1, 1]
        g = ax.pcolor(Dzs > 2.5, vmin=0, vmax=1)
        cb = plt.colorbar(g)
        cb.set_label('Decoding Performance (z-scored)')
        ax.set_xticks(np.arange(len(labels_str)) + 0.5)
        ax.set_yticks(np.arange(len(labels_str)) + 0.5)
        ax.set_xticklabels(np.asarray(labels_str)[idx], rotation=45, ha='right', va='top')
        ax.set_yticklabels(np.asarray(labels_str)[idx], rotation=0, ha='right', va='center')
        ax.set_title('Decoding performance (z-score, sorted)')

        if plot_mds:
            # Perform Multi-Dimensional Scaling
            mds = MDS(n_components=3, dissimilarity='precomputed')
            positions = mds.fit_transform(D)

            # Plotting
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for i, l in enumerate(labels_str):
                if l[0] == '2' and l[3] == 'W':
                    color = 'red'
                elif l[0] == '2' and l[3] == 'S':
                    color = pltcolors[3]
                elif l[0] == '8' and l[3] == 'W':
                    color = 'blue'
                elif l[0] == '8' and l[3] == 'S':
                    color = pltcolors[0]
                else:
                    color = 'k'
                ax.scatter(positions[i, 0], positions[i, 1], positions[i, 2], marker='$' + l + '$', s=1200, c=color)

        return D, Dz, labels_str, precomputed_perfs, f, axs

    return D, Dz, precomputed_perfs, labels_str


def independent_conditions_PS(conditioned_trials, n_neurons=None, z_threshold=2.5, perf_threshold=None,
                              merging_iteration=0, merged_labels=None,
                              plot_matrix=False, region='', folder='',
                              all_Ds=None, all_Dsz=None, precomputed_perfs=None,
                              **decoding_params):
    print(merged_labels)
    if all_Ds is None:
        all_Ds = []
    if all_Dsz is None:
        all_Dsz = []

    if plot_matrix:
        D, Dz, label, pcp, f, axs = decoding_matrix_PS(conditioned_trials, n_neurons=n_neurons,
                                                       perf_threshold=perf_threshold, plot_matrix=plot_matrix,
                                                       precomputed_perfs=precomputed_perfs,
                                                       **decoding_params)
        f.savefig(f'./plots/IBL/{folder}/IC_D/{region}_{merging_iteration}.pdf')
        plt.close(f)
    else:
        D, Dz, label, pcp = decoding_matrix_PS(conditioned_trials, n_neurons=n_neurons,
                                               perf_threshold=perf_threshold, plot_matrix=plot_matrix,
                                               precomputed_perfs=precomputed_perfs,
                                               **decoding_params)
    precomputed_perfs = pcp
    all_Ds.append(D)
    all_Dsz.append(Dz)

    if merged_labels is None:
        merged_labels = {}
    # identify those variables that are non decodable from each other
    if perf_threshold is not None:
        adjacency_matrix = D < perf_threshold
    else:
        adjacency_matrix = Dz < z_threshold

    N = {
        i: set(num for num, j in enumerate(row) if j)
        for i, row in enumerate(adjacency_matrix)
    }

    def BronKerbosch1(P, R=None, X=None):
        P = set(P)
        R = set() if R is None else R
        X = set() if X is None else X
        if not P and not X:
            yield R
        while P:
            v = P.pop()
            yield from BronKerbosch1(
                P=P.intersection(N[v]), R=R.union([v]), X=X.intersection(N[v]))
            X.add(v)

    P = N.keys()
    cliques = list(BronKerbosch1(P))
    ic = np.argmax([len(c) for c in cliques])
    max_clique = cliques[ic]

    if len(max_clique) == 1:
        clusters = merge_dict_values(merged_labels)
        reduced_labels = {}
        for k in conditioned_trials[0]:
            if ''.join(k) in clusters.keys():
                reduced_labels[''.join(k)] = clusters[''.join(k)]
            else:
                reduced_labels[''.join(k)] = [''.join(k)]
        return all_Ds, all_Dsz, reduced_labels, conditioned_trials

    merge_key = ('m', 'r', 'g', str(merging_iteration))
    new_conditioned_trials = []
    for n in range(len(conditioned_trials)):
        X = conditioned_trials[n]
        new_X = {merge_key: []}
        merged_label = []
        for i, key in enumerate(X.keys()):
            if i in max_clique:
                new_X[merge_key].append(X[key])
                merged_label.append(''.join(key))
            else:
                new_X[key] = X[key]
        new_X[merge_key] = np.vstack(new_X[merge_key])
        merged_labels[''.join(merge_key)] = merged_label
        new_conditioned_trials.append(new_X)

    # iteration
    merging_iteration += 1
    return independent_conditions_PS(new_conditioned_trials, n_neurons, z_threshold, perf_threshold, merging_iteration,
                                     merged_labels, plot_matrix, region, folder, all_Ds, all_Dsz, precomputed_perfs,
                                     **decoding_params)


def tune_noise(region, trials, L, P, synthetic_params, decoding_params, IBL_params):
    mkdir('./datasets/IBL/tune_noise')
    cachepath = f'./datasets/IBL/tune_noise/{region}-{L}-{P}-{parhash(synthetic_params)}-{parhash(decoding_params)}-{parhash(IBL_params)}.pck'
    if os.path.exists(cachepath):
        best_sigma = pickle.load(open(cachepath, 'rb'))
        return best_sigma

    cachepath = f'./datasets/IBL/tune_noise/{region}-D-{parhash(decoding_params)}-{parhash(IBL_params)}.pck'
    if os.path.exists(cachepath):
        D = pickle.load(open(cachepath, 'rb'))
    else:
        D, Dz, precomputed_perfs, labels_str = decoding_matrix_PS(trials,
                                                                  plot_matrix=False,
                                                                  perf_threshold=0.666,
                                                                  **decoding_params)
    target = np.nanmean(D[D>0])

    best_sigma = 0.1
    for sigma in np.linspace(0.1, 10.0, 100):
        synthetic_params['sigma'] = sigma
        trials_s = generate_latent_representations(L=L, P=P, **synthetic_params)
        D_s, Dz, precomputed_perfs, labels_str = decoding_matrix_PS(trials_s, plot_matrix=False,
                                                                    perf_threshold=0.666, **decoding_params)
        print(sigma, np.nanmean(D_s[D_s > 0]))
        if np.nanmean(D_s[D_s > 0]) < target:
            break
        best_sigma = sigma
    pickle.dump(best_sigma, open(cachepath, 'wb'))
    return best_sigma


def key_scatter(res, key1, key2, hierarchy, hue='region', ax=None, colors=None, corr='reg', linecolor='k'):
    figflag = False

    if ax is None:
        f, ax = plt.subplots(figsize=(4, 4))
        figflag = True
        sns.despine(ax=ax)
    ax.set_xlabel(key1)
    ax.set_ylabel(key2)
    xs = []
    ys = []
    if colors is None:
        colors = {}
        for key in hierarchy:
            for v in hierarchy[key]:
                colors[v] = key

        for i in range(len(res['region'])):
            if res['region'][i] in colors:
                if np.isnan(res[key1][i] + res[key2][i]) == 0:
                    ax.scatter(res[key1][i], res[key2][i], color=pltcolors[colors[res['region'][i]]])
                    ax.text(res[key1][i], res[key2][i], ' %s' % res[hue][i], color=pltcolors[colors[res['region'][i]]])
                    xs.append(res[key1][i])
                    ys.append(res[key2][i])
    else:
        for i in range(len(res['region'])):
            if np.isnan(res[key1][i] + res[key2][i]) == 0:
                ax.scatter(res[key1][i], res[key2][i], facecolors=colors[i], alpha=0.7, edgecolors='k', linewidth=1)
                ax.text(res[key1][i], res[key2][i], ' %s' % res[hue][i], color='k', fontsize=9)
                xs.append(res[key1][i])
                ys.append(res[key2][i])
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if corr == 'p':
        s, p = spearnan(xs, ys)
        ax.set_title(p_to_text(p), fontsize=9)
    elif corr == 'line':
        corrfunc(xs, ys, ax=ax, plot=True)
    elif corr == 'reg' or corr == 1:
        sns.regplot(x=xs, y=ys, ax=ax, color=linecolor, marker='')
        corrfunc(xs, ys, ax=ax, plot=False)
    if figflag:
        return f, ax
    else:
        return xs.shape[0]


def csd_plot(x, t0, t1, ax=None, label=None, norm=True, **kwargs):
    ts = np.linspace(t0, t1, 100)
    y = []
    for t in ts:
        y.append(np.nanmean(x < t))
    if ax is None:
        f, ax = plt.subplots(figsize=(4, 4))
    if norm:
        ax.plot(np.linspace(0, 1, 100), y, label=label, alpha=0.7, **kwargs)
        ax.set_xlim([0, 1])
    else:
        ax.plot(ts, y, label=label, alpha=0.7, **kwargs)
        ax.set_xlim([t0, t1])
    ax.set_ylim([0, 1])
    ax.legend(fontsize=10)
    return ts, np.asarray(y), ax


def generate_latent_vectors(L):
    return np.asarray(list(itertools.product([0, 1], repeat=L))) * 2 - 1


def generate_latent_representations(L, alpha, sigma, T=100, N=400, P=None, weights=None, visualize=False):
    if P is None:
        v = generate_latent_vectors(L)
        P = 2 ** L
        labels = [''.join(x) for x in ((v + 1) / 2).astype(int).astype(str)]
    else:
        v = np.random.randn(P, L)  # uniform in [-1, 1] L-cube
        labels = [f'{i}' for i in range(P)]

    if weights is not None:
        v = v * weights

    U = np.random.randn(L, N)
    trials = {}

    for i in range(P):
        trials[labels[i]] = np.dot(v[i], U) + np.random.randn(N) * alpha * 2 * np.sqrt(L) \
                            + np.random.randn(T, N) * sigma * np.sqrt(L) * np.sqrt(2 + 2. * alpha ** 2.)
    if visualize:
        X = np.vstack(list(trials.values()))
        y = np.repeat(labels, T)
        visualize_representations_PCA(X, dim=3, labels=y)
    return trials
