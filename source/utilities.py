import numpy as np

from .imports import *


def linenull(ax, null=0.5, direction='h'):
    if direction == 'h':
        ax.axhline(null, linestyle='--', color='k', alpha=0.5, linewidth=2)
    if direction == 'v':
        ax.axvline(null, linestyle='--', color='k', alpha=0.5, linewidth=2)


def n_neurons(trials):
    n = 0
    for condition in trials:
        if len(trials[condition]):
            if trials[condition].shape[1] > n:
                n = trials[condition].shape[1]
    return n


def spearnan(x, y):
    from scipy.stats import spearmanr
    nanmask = (np.isnan(x) == 0) & (np.isnan(y) == 0)
    x = np.asarray(x)[nanmask]
    y = np.asarray(y)[nanmask]
    return spearmanr(x, y)


def rotate_rasters(trials, R=None, mode='random'):
    n_neurons = 0
    for key in trials:
        if len(trials[key]) and key != '_meta_':
            n_neurons = trials[key].shape[1]
            break
    if n_neurons > 1:
        if R is None:
            R = special_ortho_group.rvs(n_neurons)
        rotated_rasters = {}
        for key in trials:
            if key != '_meta_' and len(trials[key]):
                if mode == 'random':
                    rotated_rasters[key] = np.dot(trials[key], R)
                if mode == 'index_shuffle':
                    new_index = np.arange(n_neurons)
                    np.random.shuffle(new_index)
                    rotated_rasters[key] = trials[key][:, new_index]
            else:
                rotated_rasters[key] = trials[key]
        return rotated_rasters
    else:
        return trials


def shuffle_rasters(rasters):
    # shuffled_rasters = {}
    # for key in rasters:
    #     X = rasters[key]
    #     n_data, n_neurons = X.shape
    #     idx_y = np.arange(n_neurons)
    #     np.random.shuffle(idx_y)
    #     shuffled_rasters[key] = X[:, idx_y]
    X = np.vstack([rasters[key] for key in rasters])
    n_data, n_neurons = X.shape
    idx_x = np.arange(n_data)
    idx_y = np.arange(n_neurons)
    for i in range(n_neurons):
        np.random.shuffle(idx_x)
        X[:, i] = X[idx_x, i]
    for i in range(n_data):
        np.random.shuffle(idx_y)
        X[i, :] = X[i, idx_y]
    t0 = 0
    shuffled_rasters = {}
    for key in rasters:
        n_data = rasters[key].shape[0]
        shuffled_rasters[key] = X[t0:t0+n_data, :]
        t0 += n_data
    return shuffled_rasters


def z_score_rasters(rasters):
    X = np.vstack([rasters[key] for key in rasters])
    n_data, n_neurons = X.shape

    for i in range(n_neurons):
        mean = np.nanmean(X[:, i])
        std = np.nanstd(X[:, i])
        if std:
            X[:, i] = (X[:, i] - mean) / std
        else:
            X[:, i] = np.zeros(n_data)
    t0 = 0
    zscored_rasters = {}
    for key in rasters:
        n_data = rasters[key].shape[0]
        zscored_rasters[key] = X[t0:t0 + n_data, :]
        t0 += n_data
    return zscored_rasters


def z_score_raster(X):
    n_data, n_neurons = X.shape

    for i in range(n_neurons):
        mean = np.nanmean(X[:, i])
        std = np.nanstd(X[:, i])
        if std:
            X[:, i] = (X[:, i] - mean) / std
        else:
            X[:, i] = np.zeros(n_data)
    return X


# Visualize geometry in 3D


def plot_geometry(means, names, ax):
    for i in range(len(names)):
        ax.scatter(means[i, 0], means[i, 1], means[i, 2], alpha=0.7,
                   marker='$%s$' % names[i],
                   s=100, color=pltcolors[i], edgecolor='w', linewidths=5)
        ax.scatter(means[i, 0], means[i, 1], means[i, 2], alpha=0.7,
                   marker='$%s$' % names[i],
                   s=100, color=pltcolors[i])
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if hamming_distance(names[i], names[j]) == 1:
                ax.plot([means[i][0], means[j][0]], [means[i][1], means[j][1]],
                        [means[i][2], means[j][2]], linestyle='-', linewidth=2, color='k',
                        alpha=0.7)
            elif hamming_distance(names[i], names[j]) == 2:
                ax.plot([means[i][0], means[j][0]], [means[i][1], means[j][1]],
                        [means[i][2], means[j][2]], linestyle='--', color='k', alpha=0.3)


# find all dichotomies using the power set

def dichotomy_to_key(dic):
    return '_'.join(dic[0]) + '_v_' + '_'.join(dic[1])


def all_dichotomies(conditions, only_complete=True, only_balanced=False):
    powerset = list(chain.from_iterable(combinations(conditions, r) for r in range(1, len(conditions))))
    dichotomies = {}
    for i in range(len(powerset)):
        for j in range(i + 1, len(powerset)):
            if (len(np.unique(powerset[i] + powerset[j])) == len(conditions)) or (only_complete == False):
                if len(powerset[i] + powerset[j]) == len(conditions):
                    if len(set(powerset[i]).intersection(set(powerset[j]))) == 0:
                        dic = [list(powerset[i]), list(powerset[j])]
                        dichotomies[dichotomy_to_key(dic)] = dic
    if only_balanced:
        dichotomies = {key: dichotomies[key] for key in dichotomies if len(dichotomies[key][0]) == len(dichotomies[key][1])}
    return dichotomies


def balanced_dichotomies(keys):
    """
    Generate all non-redundant balanced dichotomies (up to class label swapping).
    Each dichotomy is a pair of lists: [[class1], [class2]], with class1 < class2 lexically.

    Parameters:
        keys (list): A list of 2P unique elements

    Returns:
        list of list pairs: Each dichotomy is [[class1 elements], [class2 elements]]
    """
    N = len(keys)
    # assert N % 2 == 0, "Length of keys must be even"
    P = int(N // 2)

    all_dichotomies = []
    key_indices = list(range(N))

    for pos_indices in itertools.combinations(key_indices, P):
        class1 = [keys[i] for i in pos_indices]
        class2 = [keys[i] for i in key_indices if i not in pos_indices]

        # Enforce ordering to avoid mirrored duplicates
        if ((N % 2) == 0) and (class1 > class2):
            continue  # skip mirror
        all_dichotomies.append([class1, class2])

    return all_dichotomies


def dichotomies(keys, only_complete=False, only_balanced=False, only_pairs=False, only_semantic=False):
    if only_pairs:
        dics = {}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                dics[f'{i}_{j}'] = [[keys[i]], [keys[j]]]
    elif only_semantic:
        dics = semantic_dichotomies(keys)
    else:
        dics = all_dichotomies(keys, only_complete=only_complete,
                               only_balanced=only_balanced)
    return dics


def semantic_dichotomies(conditions):
    if len(conditions) == 4:
        return {'A': [['00', '01'], ['10', '11']], 'B': [['00', '10'], ['01', '11']]}
    if len(conditions) == 8:
        dics = {
            'A': [['000', '001', '010', '011'], ['100', '101', '110', '111']],
            'B': [['000', '001', '100', '101'], ['010', '011', '110', '111']],
            'C': [['000', '010', '100', '110'], ['001', '011', '101', '111']],
        }
        return dics


def dic_to_vec(dic):
    vec = []
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                if '%u%u%u' % (i, j, k) in dic[0]:
                    vec.append(-1)
                if '%u%u%u' % (i, j, k) in dic[1]:
                    vec.append(1)
    return np.asarray(vec)


# Utilities


def CT_to_X(conditioned_trials, zscore=True):
    if type(conditioned_trials) != list:
        CT = [conditioned_trials]
    else:
        CT = conditioned_trials
    X = np.hstack([np.vstack([np.nanmean(x, 0) for x in c.values()]) for c in CT])
    if zscore:
        for i in range(X.shape[1]):
            x = X[:, i]
            x = (x - np.nanmean(x)) / np.nanstd(x)
            X[:, i] = x
    return X


def zscore_CT(conditioned_trials):
    zscored_conditioned_trials = []
    for c in conditioned_trials:
        zc = {}
        X = np.vstack([np.nanmean(x, 0) for x in c.values()])
        mean = np.nanmean(X, 0)
        std = np.nanstd(X, 0)
        std[std == 0] = 1

        for key in c:
            zc[key] = (c[key] - mean) / std
        zscored_conditioned_trials.append(zc)
    return zscored_conditioned_trials


def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g


def sort_with_nan(D):
    valid_i = np.sum(np.isnan(D), axis=0) < D.shape[0]
    valid_j = np.sum(np.isnan(D), axis=1) < D.shape[1]

    # Extract the valid submatrix
    X = D[valid_i, :]
    X = X[:, valid_j]
    X, idx = sort_matrix(X)

    # Reinsert the sorted valid submatrix into the original matrix shape
    sorted_matrix = np.empty_like(D)

    sorted_matrix[:np.sum(valid_i), :np.sum(valid_j)] = X
    return sorted_matrix


def sort_matrix(D):
    def get_sim_mat2(sm):
        n = sm.shape[0]
        return 1 / (np.linalg.norm(sm[:, np.newaxis] - sm[np.newaxis], axis=-1) + 1 / n)

    def argsort_sim_mat(sm):
        idx = [np.argmax(np.sum(sm, axis=1))]  # a
        for i in range(1, len(sm)):
            sm_i = sm[idx[-1]].copy()
            sm_i[idx] = -1
            idx.append(np.argmax(sm_i))  # b
        return np.array(idx)

    sim_mat2 = get_sim_mat2(D)
    idx = argsort_sim_mat(sim_mat2)

    return D[idx, :][:, idx], idx


def merge_dict_values(d):
    def resolve_key(key, resolved, unresolved):
        if key in resolved:
            return resolved[key]

        if key in unresolved:
            raise ValueError(f"Circular reference detected: {key}")

        unresolved.add(key)
        value = d[key]

        if isinstance(value, list):
            merged_value = []
            for item in value:
                if item in d:
                    merged_value.extend(resolve_key(item, resolved, unresolved))
                else:
                    merged_value.append(item)
            resolved[key] = merged_value
        else:
            resolved[key] = value

        unresolved.remove(key)
        return resolved[key]

    resolved = {}
    for key in d:
        resolve_key(key, resolved, set())

    return resolved


def parhash(params):
    name = '_'.join([k + '=%s' % params[k] for k in params.keys()])
    s = hashlib.md5(bytes(name, 'utf-8')).hexdigest()
    return name, s[:8]


def write_params(dir, params):
    with open(dir+'/params.txt', 'w') as f:
        for k in params:
            f.write(f'{k}: {params[k]}\n')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def p_to_text(p):
    if p < 0.0001:
        return '*** P=%.1e' % p
    if p < 0.001:
        return '*** P=%.4f' % p
    if p < 0.002:
        return '** P=%.4f' % p
    if p < 0.01:
        return '** P=%.3f' % p
    if p < 0.02:
        return '* P=%.3f' % p
    if p < 0.05:
        return '* P=%.3f' % p
    if p >= 0.05:
        return 'ns P=%.2f' % p


def bonferroni_z(p, m):
    """
    Compute the Bonferroni-corrected significance level and corresponding z-scores.

    Parameters:
    p (float): Initial significance level (e.g., 0.05)
    m (int): Number of tests

    Returns:
    dict: Corrected alpha, one-tailed z-score, and two-tailed z-score
    """
    alpha_corrected = p / m
    z_score_one_tailed = scipy.stats.norm.ppf(1 - alpha_corrected)
    z_score_two_tailed = scipy.stats.norm.ppf(1 - alpha_corrected / 2)

    return z_score_one_tailed, z_score_two_tailed


def nullplots(ax, n, invert_sign=False):
    z2, z1 = bonferroni_z(0.05, n)
    if invert_sign:
        z1 = -z1
        z2 = -z2
    linenull(ax, z1)
    xs = ax.get_xlim()
    ys = ax.get_ylim()
    if invert_sign:
        ax.fill_between(xs, z1, ys[1], color='k', alpha=0.05, zorder=-100)
    else:
        ax.fill_between(xs, ys[0], z1, color='k', alpha=0.05, zorder=-100)
    ax.set_xlim(xs)
    ax.set_ylim(ys)


def equalize_axs(ax, plot_xy=False):
    ys = ax.get_ylim()
    xs = ax.get_xlim()
    minv = np.min([xs[0], ys[0]])
    maxv = np.max([ys[1], xs[1]])
    if plot_xy:
        ax.plot([minv, maxv], [minv, maxv], '--k')
    ax.set_xlim([minv, maxv])
    ax.set_ylim([minv, maxv])


def linear_regression_confidence(x, y, confidence=0.95):
    n = len(x)
    nanmask = (~np.isnan(x)) & (~np.isnan(y))
    x = x[nanmask]
    y = y[nanmask]
    # Compute the optimal slope a
    a = np.sum(x * y) / np.sum(x ** 2)

    # Compute predicted values
    y_pred = a * x

    # Compute residual sum of squares
    residuals = y - y_pred
    rss = np.sum(residuals ** 2)

    # Compute standard error of the slope
    se_a = np.sqrt(rss / ((n - 1) * np.sum(x ** 2)))

    # Compute t-critical value
    t_crit = scipy.stats.t.ppf((1 + confidence) / 2, df=n - 1)

    # Compute confidence interval
    conf_interval = (a - t_crit * se_a, a + t_crit * se_a)

    return a, se_a, conf_interval


def visualize_data_vs_null(data, null, value, ax=None):
    # computing the P value of the z-score
    from scipy.stats import norm
    null_mean = np.nanmean(null)
    z = (data - null_mean) / np.nanstd(null)
    p = norm.sf(abs(z))
    null = np.asarray(null)

    def p_to_ast(p):
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        if p >= 0.05:
            return 'ns'

    # visualizing
    if ax is None:
        f, ax = plt.subplots(figsize=(6, 3))
    nullkde = null[np.isnan(null) == 0]
    nullkde = nullkde[np.isinf(nullkde) == 0]
    if len(nullkde):
        kde = scipy.stats.gaussian_kde(nullkde)
        null_x = np.linspace(np.nanmean(null) - 5 * np.nanstd(null), np.nanmean(null) + 5 * np.nanstd(null), 100)
        null_y = kde(null_x)
        ax.plot(null_x, null_y, color='k', alpha=0.5)
        ax.fill_between(null_x, null_y, color='k', alpha=0.4)
        ax.text(null_mean, np.max(null_y) * 0.1, 'null', ha='center', color='w', fontweight='bold')
        sns.despine(ax=ax)
        ax.plot([data, data], [0, np.max(null_y)], color='red', linewidth=3)
        ax.text(data, np.max(null_y) * 1.05, 'data', ha='left', color=pltcolors[3])
        ax.set_xlabel(value)
    if data < np.nanmean(null):
        ax.text(0.85, 0.85, '%s\nz=%.1f\nP=%.1e' % (p_to_ast(p), z, p), ha='center', transform=ax.transAxes)
    else:
        ax.text(0.15, 0.85, '%s\nz=%.1f\nP=%.1e' % (p_to_ast(p), z, p), ha='center', transform=ax.transAxes)

    # ax.plot(null, np.zeros(len(null)), linestyle='', marker='|', color='k')
    _ = ax.plot([null_mean, null_mean], [0, kde(null_mean)[0]], color='k', linestyle='--')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return z, p


def visualize_population_vs_null(data, null, value, label='', color='k', ax=None):
    # computing the P value of the z-score
    from scipy.stats import norm
    data_mean = np.nanmean(data)
    t, p = ttest_1samp(data[np.isnan(data) == 0], null)

    def p_to_ast(p):
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        if p >= 0.05:
            return 'ns'

    # visualizing
    if ax is None:
        f, ax = plt.subplots(figsize=(6, 3))
    kde = data[np.isnan(data) == 0]
    kde = kde[np.isinf(kde) == 0]
    if len(kde):
        kde = scipy.stats.gaussian_kde(kde)
        data_x = np.linspace(np.nanmean(data) - 5 * np.nanstd(data), np.nanmean(data) + 5 * np.nanstd(data), 100)
        data_y = kde(data_x)
        ax.plot(data_x, data_y, color=color, alpha=0.5)
        ax.fill_between(data_x, data_y, color=color, alpha=0.4)
        ax.text(data_mean, kde(data_mean) * 1.05, label+'\n'+p_to_text(p), ha='left', color=color, fontweight='bold')
        print(data_mean, kde(data_mean))
        ax.plot([data_mean, data_mean], [0, kde(data_mean)[0]], color=color, linewidth=2)
        sns.despine(ax=ax)
        ax.axvline([null], color='k', linewidth=2, linestyle='--', alpha=0.5)
        ax.set_xlabel(value)

    # ax.plot(null, np.zeros(len(null)), linestyle='', marker='|', color='k')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return p


def setup_decoding_axis(ax, labels, ylow=None, yhigh=None, null=0.5):
    ax.set_ylabel('Decoding performance')
    ax.axhline([null], linestyle='--', color='k')
    ax.set_xticks(range(len(labels)))
    if len(labels):
        ax.set_xlim([-0.5, len(labels) - 0.5])
    ax.set_xticklabels(labels, rotation=45, ha='right')
    if ylow is not None and yhigh is not None:
        for i in range(1, int((yhigh - null) / 0.1) + 1):
            ax.axhline([null + i * 0.1], linestyle='-', color='k', alpha=0.1)
            ax.axhline([null - i * 0.1], linestyle='-', color='k', alpha=0.1)
        ax.set_ylim([ylow, yhigh + 0.04])
    sns.despine(ax=ax)


def plot_perfs_null_model(perfs, perfs_nullmodel, marker='o', ylabel='Decoding performance', ax=None, shownull=False,
                          chance=0.5, setup=True, ptype='z', annotate=True, ylow=None, yhigh=None, **kwargs):
    labels = list(perfs.keys())
    pvals = {}
    if not ax:
        f, ax = plt.subplots(figsize=(1.2 * len(perfs), 4))
    if shownull == 'swarm':
        sns.swarmplot(ax=ax, data=pd.DataFrame(perfs_nullmodel, columns=labels), alpha=0.2, size=4, color='k')
    if shownull == 'violin':
        sns.violinplot(ax=ax, data=pd.DataFrame(perfs_nullmodel, columns=labels), color=[0.8, 0.8, 0.8, 0.3], bw=.25,
                       cut=0, inner=None)
    if setup:
        setup_decoding_axis(ax, labels, ylow=ylow, yhigh=yhigh, null=chance)
    ax.set_ylabel(ylabel)

    for i, l in enumerate(labels):
        if ptype == 'count':
            top_quartile = np.percentile(perfs_nullmodel[l], 95) - np.nanmean(perfs_nullmodel[l])
            low_quartile = np.nanmean(perfs_nullmodel[l]) - np.percentile(perfs_nullmodel[l], 5)
            ax.errorbar([i], np.nanmean(perfs_nullmodel[l]), yerr=np.asarray([[low_quartile, top_quartile]]).T,
                        color='k',
                        linewidth=2, capsize=5, marker='_', alpha=0.3)
            pval = np.nanmean(np.asarray(perfs_nullmodel[l]) > perfs[l])

        if ptype == 'zscore' or ptype == 'z':
            ax.errorbar([i], np.nanmean(perfs_nullmodel[l]), yerr=2 * np.nanstd(perfs_nullmodel[l]), color='k',
                        linewidth=2, capsize=5, marker='_', alpha=0.5)
            pval = z_pval(perfs[l], perfs_nullmodel[l])[1]

        ax.scatter([i], [perfs[l]], marker=marker, s=100, color=pltcolors[i], facecolor='none', linewidth=2)
        if annotate:
            ptext = p_to_text(pval).split(' ')[1]
            trans = transforms.blended_transform_factory(
                ax.transData, ax.transAxes)
            ax.text(i - 0.22, 0.0,
                    'data=%.2f\nnull=%.2f$\pm$%.2f\n%s' %
                    (perfs[l], np.nanmean(perfs_nullmodel[l]), np.nanstd(perfs_nullmodel[l]), ptext),
                    va='bottom', ha='left', fontsize=7, backgroundcolor='1.00', transform=trans)
            ax.set_xticklabels(labels, rotation=0, ha='center')

        pvals[l] = pval
        if pval < 0.05:
            ax.plot([i - 0.2, i - 0.22, i - 0.22, i - 0.2],
                    [np.nanmean(perfs_nullmodel[l]), np.nanmean(perfs_nullmodel[l]), perfs[l], perfs[l]],
                    color='k', alpha=0.5, linewidth=1.0)
            ax.text(i - 0.18, 0.5 * (np.nanmean(perfs_nullmodel[l]) + perfs[l]), p_to_ast(pval), rotation=90,
                    ha='right', va='center', fontsize=14)
    return pvals


def decode_cache(data, conditions, decodanda_params, decoding_params, savename, cache):
    decodanda_params_essential = copy.deepcopy(decodanda_params)
    if 'verbose' in decodanda_params_essential.keys():
        del decodanda_params_essential['verbose']
    if 'x_factor' in decoding_params.keys():
        x_factor = decoding_params['x_factor']
    else:
        x_factor = 1

    s, h = parhash(decoding_params)
    s1, h1 = parhash(decodanda_params_essential)
    keys = ''
    for key in conditions:
        keys += '_%s' % key

    filename = './cache/%s_%s_decoding.pck' % (savename, h + h1 + keys)
    if os.path.exists(filename) and cache:
        res, null, w, _ = pickle.load(open(filename, 'rb'))
    else:
        dec = Decodanda(x_factor * data, conditions, **decodanda_params)
        res, null = dec.decode(**decoding_params)
        dec = Decodanda(data, conditions, **decodanda_params)
        new_decoding_params = copy.deepcopy(decoding_params)
        new_decoding_params['nshuffles'] = 0
        dec.decode(**new_decoding_params)
        w = dec.decoding_weights
        if cache:
            pickle.dump([res, null, w, [decoding_params, decodanda_params]], open(filename, 'wb'))
    return res, null, w


def CCGP_cache(data, conditions, decodanda_params, decoding_params, savename, cache):
    decodanda_params_essential = copy.deepcopy(decodanda_params)
    if 'verbose' in decodanda_params_essential.keys():
        del decodanda_params_essential['verbose']
    if 'x_factor' in decoding_params.keys():
        x_factor = decoding_params['x_factor']
    else:
        x_factor = 1

    s, h = parhash(decoding_params)
    s1, h1 = parhash(decodanda_params_essential)
    keys = ''
    for key in conditions:
        keys += '_%s' % key

    filename = './cache/%s_%s_ccgp.pck' % (savename, h + h1 + keys)
    if os.path.exists(filename) and cache:
        res, null, _ = pickle.load(open(filename, 'rb'))
    else:
        dec = Decodanda(x_factor * data, conditions, **decodanda_params)
        res, null = dec.CCGP(ntrials=decoding_params['cross_validations'], **decoding_params)
        if cache:
            pickle.dump([res, null, [decoding_params, decodanda_params]], open(filename, 'wb'))
    return res, null


class auc_computer(object):
    def __init__(self, rasterA, rasterB):
        self.rasterA = rasterA
        self.rasterB = rasterB

    def __call__(self, i):
        xA = self.rasterA[:, i]
        xB = self.rasterB[:, i]
        y = np.hstack([np.repeat(0, len(xA)), np.repeat(1, len(xB))])
        x = np.hstack([xA, xB])
        fpr, tpr, thresholds = roc_curve(y, x)
        auc_data = auc(fpr, tpr)
        return auc_data


class AUCoder(BaseEstimator):
    def __init__(self, rotated=False, n_neurons=0):
        self.coef_ = []
        self.R = None
        self.n_neurons = n_neurons
        self.rotated = rotated
        self.nullcoef_ = []
        # self.pool = Pool()
        if self.rotated:
            self.sample_rotation_matrix(self.n_neurons)

    def fit(self, raster, labels):
        t0 = time.time()
        [lA, lB] = np.unique(labels)
        rasterA = raster[np.asarray(labels) == lA]
        rasterB = raster[np.asarray(labels) == lB]
        self.coef_ = np.zeros(raster.shape[1])
        if self.R is not None:
            rasterA = np.dot(rasterA, self.R)
            rasterB = np.dot(rasterB, self.R)
        # print('computing coefficients')
        # t0 = time.time()
        # self.coef_ = self.pool.map(auc_computer(rasterA, rasterB), range(raster.shape[1]))
        # print(time.time() - t0)

        self.coef_ = np.zeros(raster.shape[1])
        for i in range(raster.shape[1]):
            xA = rasterA[:, i]
            xB = rasterB[:, i]
            y = np.hstack([np.repeat(0, len(xA)), np.repeat(1, len(xB))])
            x = np.hstack([xA, xB])
            fpr, tpr, thresholds = roc_curve(y, x)
            self.coef_[i] = auc(fpr, tpr)
        # print(time.time() - t0)

        # print('-> computed coefficients')

    def predict(self, raster):
        return np.ones(len(raster))

    def score(self, raster, labels):
        return np.nanmean(self.coef_)

    def sample_rotation_matrix(self, n_neurons):
        self.R = special_ortho_group.rvs(n_neurons)


def annotate_perfs(perfs, ax, pvals, null=0.5):
    labels = list(perfs.keys())
    for x, key in enumerate(labels):
        if np.nanmean(perfs[key]) > null:
            linepadding = 0.01
        else:
            linepadding = -0.01
        ax.plot([x - 0.15, x - 0.15, x - 0.12], [null + linepadding, np.nanmean(perfs[key]), np.nanmean(perfs[key])],
                color='k',
                alpha=0.5, linewidth=1)
        ax.text(x - 0.14, 0.5 * (np.nanmean(perfs[key]) + null), p_to_ast(pvals[key]), rotation=90, ha='right',
                va='center',
                fontsize=14)

        ptext = p_to_text(pvals[key])
        trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
        ax.text(x - 0.22, 0.0,
                'data=%.2f$\pm$%.2f\n%s' %
                (np.nanmean(perfs[key]), np.nanstd(perfs[key]), ptext),
                va='bottom', ha='left', fontsize=7, backgroundcolor='1.00', transform=trans)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=0, ha='center')


def plot_perfs_swarm(performances, ax, chance=0.5, test=ttest_1samp):
    labels = list(performances.keys())

    sns.swarmplot(data=pd.DataFrame.from_dict(performances), ax=ax)
    for i in range(len(performances.keys())):
        ax.errorbar([i], np.nanmean(performances[labels[i]]), np.nanstd(performances[labels[i]]),
                    color=pltcolors[i], capsize=6, linewidth=2, alpha=0.7)
    pvals = {key: test(performances[key], chance)[1] for key in labels}
    print(pvals)
    annotate_perfs(performances, ax, pvals, null=chance)


def dic_key(dic):
    return '_'.join(dic[0]) + '_v_' + '_'.join(dic[1])


def cohend(Adata, Bdata, paired=False):
    if not paired:
        d = np.nanmean(Adata)-np.nanmean(Bdata)
        psd = np.sqrt((np.nanvar(Adata) + np.nanvar(Bdata))/2.)
    else:
        diff = np.asarray(Bdata) - np.asarray(Adata)
        d = np.nanmean(diff)
        psd = np.nanstd(diff)
    return d/psd


def paired_analysis(data_1, data_2, data_labels=None, xlabels=None, ylabel=None, ax=None, test=scipy.stats.ttest_rel, **kwargs):
    nanmask = (np.isnan(data_1) == 0) & (np.isnan(data_2) == 0)
    values_1 = np.asarray(data_1)[nanmask]
    values_2 = np.asarray(data_2)[nanmask]
    colors = pltcolors*10
    if ax is None:
        f, ax = plt.subplots(figsize=(2.5, 3.5))
    ax.set_xticks([0, 1])
    ax.set_xlim([-0.5, 1.5])
    if xlabels is not None:
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
    sns.despine(ax=ax)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    for i in range(len(values_1)):
        if np.isnan(values_1[i]) == 0 and np.isnan(values_2[i]) == 0:
            ax.plot([0, 1], [values_1[i], values_2[i]], color=colors[i], **kwargs)
    if data_labels is not None:
        names = np.asarray(data_labels)[nanmask]
        for i, m in enumerate(names):
            if np.isnan(values_1[i]) == 0 and np.isnan(values_2[i]) == 0:
                ax.text(1.1, values_2[i], m, ha='left', alpha=0.5, color=colors[i], fontsize=6)
    ax.errorbar([0, 1], [np.nanmean(values_1), np.nanmean(values_2)], [np.nanstd(values_1)/np.sqrt(len(values_1)), np.nanstd(values_2)/np.sqrt(len(values_2))],
                linewidth=3, alpha=0.5, capsize=4, color='k')

    p = test(values_1, values_2)[1]
    ys = [np.min([np.min(values_1), np.min(values_2)]), np.max([np.max(values_1), np.max(values_2)])]
    dy = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
    x1 = 0
    x2 = 1
    y = ys[1] + dy
    dx = 0
    ax.plot([x1 + dx, x1 + dx, x2 - dx, x2 - dx], [y, y + dy / 2, y + dy / 2, y], 'k')
    ax.text((x1 + x2) / 2., y + 2 * dy, p_to_text(p), ha='center', va='bottom', color='k', fontsize=12)

    print("Paired t-test p=%.2e, cohen's d=%.2f" % (p, cohend(values_1, values_2, True)))
    return values_1, values_2, p


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.dot(v1_u, v2_u))


def compute_angles(X):
    G = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            v1 = X[i]
            v2 = X[j]
            G[i, j] = angle_between(v1, v2)
            G[j, i] = G[i, j]
    return G


def clustering_inertia(X, clusters, nshuffles=100):
    inertias = {}
    inertias_null = {}
    clusters_shuffled = np.copy(clusters)

    for k in np.unique(clusters):
        x = X[clusters == k]
        ine = cdist(x, x)[np.triu_indices(x.shape[0])]
        inertias[k] = np.nanmean(ine)
        inertias_null[k] = []
        for n in range(nshuffles):
            np.random.shuffle(clusters_shuffled)
            x = X[clusters_shuffled == k]
            ine = cdist(x, x)[np.triu_indices(x.shape[0])]
            inertias_null[k].append(np.nanmean(ine))
    if nshuffles:
        return inertias, inertias_null
    else:
        return inertias


# Visualization tools

def categorical_bars(data, null=0.5, ymin=0.0, maxval=1.0,
                     quantity='', ax=None, cmap='PiYG',
                     vmax=None, color_scheme=None, pvals=None):
    axflag = False
    if ax is None:
        f, ax = plt.subplots(figsize=(14, 3))
        axflag = True
    if vmax is None:
        vmax = maxval
    ax.set_ylabel(quantity)
    sns.despine(ax=ax)
    labels = np.asarray(list(data.keys()))
    values = np.asarray(list(data.values()))
    pvalues = np.asarray(list(pvals.values()))
    cm = plt.get_cmap(cmap)
    if color_scheme is None:
        colors = cm(0.5 + np.abs((values - null)) / (vmax - null))
    else:
        colors = []
        for i in range(len(values)):
            a = np.min([1, np.max([0, np.abs(values[i] - null) / (vmax - null)])])
            if labels[i] in color_scheme:
                color = color_scheme[labels[i]] * a + np.ones(4) * (1 - a)
                color[3] = 1.0
                colors.append(color)
            else:
                color = np.ones(4) * (1 - 0.8 * a)
                color[3] = 1.0
                colors.append(color)

    x = np.arange(len(values))
    ax.bar(x, values - null, color=colors, bottom=null)
    ax.axhline(null, color='k', alpha=0.3, linewidth=2, linestyle='--')
    if pvals is not None:
        for i in range(len(values)):
            if values[i] > null:
                ax.text(x[i], values[i] + 0.01, '%.2f\n%s' % (values[i], p_to_ast(pvalues[i])), ha='center', va='bottom', fontsize=6)
            if values[i] <= null:
                ax.text(x[i], values[i] - 0.01, '%.2f\n%s' % (values[i], p_to_ast(pvalues[i])), ha='center', va='top', fontsize=6)
    else:
        for i in range(len(values)):
            if values[i] > null:
                ax.text(x[i], values[i] + 0.01, '%.2f' % values[i], ha='center', va='bottom', fontsize=6)
            if values[i] <= null:
                ax.text(x[i], values[i] - 0.01, '%.2f' % values[i], ha='center', va='top', fontsize=6)

    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=60, fontsize=8, ha='right', va='top')
    ax.set_xlim([-1, len(labels)])
    ax.set_ylim([ymin, maxval])
    if axflag:
        return f, ax
    else:
        return ax


def sorted_bars(data, null=0.5, ymin=0.0, maxval=1.0,
                quantity='', ax=None, cmap='PiYG',
                vmax=None, color_scheme=None):
    axflag = False
    if ax is None:
        f, ax = plt.subplots(figsize=(3, 3))
        axflag = True
    if vmax is None:
        vmax = maxval
    ax.set_ylabel(quantity)
    sns.despine(ax=ax)
    labels = np.asarray(list(data.keys()))
    values = np.asarray(list(data.values()))
    idx = np.flip(np.argsort(values))
    labels = labels[idx]
    values = values[idx]
    cm = plt.get_cmap(cmap)
    if color_scheme is None:
        colors = cm(0.5 + (values - null) / (vmax - null))
    else:
        colors = []
        for i in range(len(values)):
            a = np.min([1, np.max([0, (values[i] - null) / (vmax - null)])])
            if labels[i] in color_scheme:
                colors.append(color_scheme[labels[i]])
            else:
                colors.append(np.ones(4) * 0.333)

    x = np.arange(len(values))
    ax.bar(x, values - null, color=colors, bottom=null)
    ax.plot(x, values, color='k', linewidth=2, alpha=0.5)
    ax.axhline(null, color='k', alpha=0.3, linewidth=2, linestyle='--')
    ax.set_xlim([-3, len(labels) + 2])
    ax.set_ylim([ymin, maxval])
    if axflag:
        return f, ax
    else:
        return ax


def divide_into_bins(times, bins, x=None, bin_function=np.nanmean):
    # Ensure t and x are numpy arrays
    times = np.array(times)
    if x is None:
        x = np.ones(len(times))
    else:
        x = np.array(x)

    # Initialize an array to store the average values
    averages = np.zeros(len(bins) - 1)

    # Iterate over the time bins
    for i in range(len(bins) - 1):
        # Find indices of t that fall within the current time bin
        indices = np.where((times >= bins[i]) & (times < bins[i + 1]))[0]

        # Calculate the average of x in this time bin
        if len(indices) > 0:
            averages[i] = bin_function(x[indices])

    return averages


def save_rotating_3d_scatter(X, filename='rotation.gif', frames=90, interval=50, **kwargs):
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[0], X[1], X[2], **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.set_facecolor((0,0,0,0))  # Set axis background to transparent

    def update(frame):
        ax.view_init(elev=10, azim=frame * 4)
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    ani.save(filename, writer='ffmpeg')
    plt.close(fig)
    print(f"Animation saved as {filename}")
