import matplotlib.pyplot as plt
import numpy as np

from IBL_settings import *
from source.analysis.Dimensionality import *
from source.analysis.Clustering import *

H = pd.read_csv('./datasets/IBL/area_list.csv', header=None).to_dict()[0]
h = {}
for key in H:
    h[H[key]] = key
reduced_CT = pickle.load(open('./datasets/IBL/reduced_CT.pck', 'rb'))
cortical_regions = list(reduced_CT.keys())
[allperfs, allperfs_null] = pickle.load(open('./datasets/IBL/allperfs_L.pck', 'rb'))


def filter(region):
    trials = reduced_CT[region]
    keys = list(trials[0].keys())
    n = np.sum([c[keys[0]].shape[1] for c in trials])
    return n >= 50


def SD(data, null):
    return np.nanmean(np.asarray(data) > 0.75)
    # return np.nanmean(np.asarray(data) > (np.nanmean(null) + 3*np.nanstd(null)))
    # return np.nanmean(np.asarray(data) > np.nanpercentile(null, 100))


f, axs = plt.subplots(1, 6, figsize=(20, 3.5))
for i, L in enumerate([1, 2, 3, 4, 5, -1]):
    results = {'region': [], 'Separability': [], 'Position in the hierarchy': []}
    for region in allperfs:
        if filter(region):
            results['region'].append(region)
            results['Position in the hierarchy'].append(h[region])
            if L in allperfs[region]:
                results['Separability'].append(SD(allperfs[region][L], allperfs_null[region][L]))
            else:
                results['Separability'].append(SD(allperfs[region][-1], allperfs_null[region][-1]))
    key_scatter(results, 'Position in the hierarchy', 'Separability', groups, corr=None,
                colors=100 * [pltcolors[0]], linecolor=pltcolors[0], ax=axs[i])
    if L>0:
        axs[i].set_title(f'Projected L={L}')
    else:
        axs[i].set_title(f'Original Data')
    axs[i].set_ylim([0, 1.05])
    axs[i].grid()
    axs[i].axhline([np.nanmean(results['Separability'])], linewidth=2.0, color='b', linestyle='--')
plt.suptitle('perf > 0.75')

f, axs = plt.subplots(4, 5, figsize=(18, 12))
i = 0
for region in allperfs:
    if filter(region):
        ax = axs.ravel()[i]
        ax.set_title(region)
        ax = axs.ravel()[i]
        Ls = list(allperfs[region].keys())[1:]
        SD_orig = SD(allperfs[region][-1], allperfs_null[region][-1])
        SD_squeezed = [SD(allperfs[region][L], allperfs_null[region][L]) for L in Ls]

        ax.plot(Ls, SD_squeezed, color='k')
        for l, L in enumerate(Ls):
            ax.plot(L, SD_squeezed[l], color=pltcolors[l + 1], marker='o')
        ax.axhline(SD_orig, color=pltcolors[0], linestyle='--', linewidth=2.0)
        ax.set_ylabel('SD threshold')
        ax.set_xticks(Ls)
        ax.set_ylim([0., 1.0])
        ax.set_xlim([0, 15])
        ax.set_xlabel('Latent dimensionality $L$')
        ax.axvline(np.log2(len(Ls)), color='g', linestyle='--')
        i += 1

f.savefig('./plots/IBL/SD_squeeze_all.pdf')






results = {
    'region': [],
    'best dim': [],
    'IC': [],
    'Teff': [],
    'PR': [],
    'Position in the hierarchy': []
}

for i, region in enumerate(list(reduced_CT.keys())):
    if filter(region):
        results['region'].append(str(region))
        print(region)
        results['best dim'].append(int(np.argmax(allmeans[region][:-1]) + 1))
        trials = reduced_CT[region]
        IC = len(trials[0].keys())
        results['IC'].append(IC)
        min_Ts = [np.sum([s[key].shape[0] for key in s]) for s in trials]
        results['Teff'].append(np.nanmean(min_Ts))
        results['PR'].append(participation_ratio(CT_to_X(trials, zscore=True)))
        results['Position in the hierarchy'].append(h[region])

key_scatter(results, 'IC', 'best dim', groups, corr='reg', colors=100 * [pltcolors[0]], linecolor=pltcolors[0])
key_scatter(results, 'PR', 'best dim', groups, corr='reg', colors=100 * [pltcolors[0]], linecolor=pltcolors[0])
key_scatter(results, 'Position in the hierarchy', 'best dim', groups, corr='reg', colors=100 * [pltcolors[0]],
            linecolor=pltcolors[0])
key_scatter(results, 'Teff', 'best dim', groups, corr='p', colors=100 * [pltcolors[0]], linecolor=pltcolors[0])
