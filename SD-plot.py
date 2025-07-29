import matplotlib
import numpy as np
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.ion()
from IBL_settings import *
from source.analysis.Dimensionality import *
from source.analysis.Clustering import *
mkdir('./plots/revisions')

# ---- Separability ---
N = 640
P = 16
k = 2
alphas = np.linspace(0.05, 2.0, 40)
sigmas = np.asarray([0.25, 0.5, 1.0, 1.5, 2.0])

f, axs = plt.subplots(1, 3, figsize=(8, 3.5))

for sigma in sigmas:
    savename = f'./datasets/IBL/SD_cache/synthetic_k={k}_N={N}_P={P}_s={sigma}_a0={alphas[0]}_af={alphas[-1]}_na={len(alphas)}.pck'
    [PRs, SDs, DPs, SSs] = pickle.load(open(savename, 'rb'))
    axs[0].plot(SSs, PRs, label=f'$\sigma={sigma:.2f}$', linewidth=2.5, alpha=0.5)
    axs[0].set_xlabel('Silhouette Score')
    axs[0].set_ylabel('PR')
    axs[0].legend(fontsize=8)

    axs[1].plot(SSs, SDs, label=f'$\sigma={sigma:.2f}$', linewidth=2.5, alpha=0.5)
    axs[1].set_xlabel('Silhouette Score')
    axs[1].set_ylabel('Separability')
    axs[1].legend(fontsize=8)

    axs[2].plot(SSs, DPs, label=f'$\sigma={sigma:.2f}$', linewidth=2.5, alpha=0.5)
    axs[2].set_xlabel('Silhouette Score')
    axs[2].set_ylabel('Mean Decoding Performance')
    axs[2].legend(fontsize=8)
f.savefig('./plots/revisions/synthetic_SS-vs-dim.pdf')


f, axs = plt.subplots(1, 3, figsize=(8, 3.5))

for sigma in sigmas:
    savename = f'./datasets/IBL/SD_cache/synthetic_k={k}_N={N}_P={P}_s={sigma}_a0={alphas[0]}_af={alphas[-1]}_na={len(alphas)}.pck'
    [PRs, SDs, DPs, SSs] = pickle.load(open(savename, 'rb'))
    axs[0].plot(alphas, PRs[1:], label=f'$\sigma={sigma:.2f}$', linewidth=2.5, alpha=0.5)
    axs[0].set_xlabel('Cluster Diversity $\delta$')
    axs[0].set_ylabel('PR')
    axs[0].legend(fontsize=8)

    axs[1].plot(alphas, SDs[1:], label=f'$\sigma={sigma:.2f}$', linewidth=2.5, alpha=0.5)
    axs[1].set_xlabel('Cluster Diversity $\delta$')
    axs[1].set_ylabel('Separability')
    axs[1].legend(fontsize=8)

    axs[2].plot(alphas, DPs[1:], label=f'$\sigma={sigma:.2f}$', linewidth=2.5, alpha=0.5)
    axs[2].set_xlabel('Cluster Diversity $\delta$')
    axs[2].set_ylabel('Mean Decoding Performance')
    axs[2].legend(fontsize=8)
f.savefig('./plots/revisions/synthetic_diversity-vs-dim.pdf')


PR = {}
SD = {}
DP = {}
for a in alphas:
    i = np.where(alphas == a)[0][0]
    PR_a = []
    SD_a = []
    DP_a = []
    for sigma in sigmas:
        savename = f'./datasets/IBL/SD_cache/synthetic_k={k}_N={N}_P={P}_s={sigma}_a0={alphas[0]}_af={alphas[-1]}_na={len(alphas)}.pck'
        [PRs, SDs, DPs, SSs] = pickle.load(open(savename, 'rb'))
        PR_a.append(PRs[i])
        SD_a.append(SDs[i])
        DP_a.append(DPs[i])
    PR[round(a, 2)] = PR_a
    SD[round(a, 2)] = SD_a
    DP[round(a, 2)] = DP_a

X_SD = np.vstack(list(SD.values()))
X_DP = np.vstack(list(DP.values()))
f, axs = plt.subplots(1, 2, figsize=(9, 4))
g = axs[0].pcolor(sigmas, alphas, X_SD)
plt.colorbar(g)
axs[0].set_title('Separability')
g = axs[1].pcolor(sigmas, alphas, X_DP, vmin=0.5, vmax=1.0)
plt.colorbar(g)
axs[1].set_title('Shattering Dim')
axs[0].set_xlabel('Trial-to-trial noise $\sigma$')
axs[0].set_ylabel('Cluster Diversity $\delta$')
axs[1].set_xlabel('Trial-to-trial noise $\sigma$')
axs[1].set_ylabel('Cluster Diversity $\delta$')
f.savefig('./plots/revisions/synthetic_phase-diagram.pdf')


f, axs = plt.subplots(1, 3, figsize=(10, 3.5))
plotalphas = [0.1, 0.2, 0.3, 0.5, 1.0]
for a in plotalphas:
    axs[0].plot(2/sigmas, PR[a], label=f'$\delta={a:.2f}$', linewidth=2.5, alpha=0.5)
    axs[0].set_xlabel('Trial-to-trial SNR $1/\sigma$')
    axs[0].set_ylabel('PR')
    axs[0].legend(fontsize=8)

    axs[1].plot(2/sigmas, SD[a], label=f'$\delta={a:.2f}$', linewidth=2.5, alpha=0.5)
    axs[1].set_xlabel('Trial-to-trial SNR $1/\sigma$')
    axs[1].set_ylabel('Separability')
    axs[1].legend(fontsize=8)

    axs[2].plot(2/sigmas, DP[a], label=f'$\delta={a:.2f}$', linewidth=2.5, alpha=0.5)
    axs[2].set_xlabel('Trial-to-trial SNR $1/\sigma$')
    axs[2].set_ylabel('Mean Decoding Performance')
    axs[2].legend(fontsize=8)
f.savefig('./plots/revisions/synthetic_noise-vs-dim.pdf')


# ---

H = pd.read_csv('./datasets/IBL/area_list.csv', header=None).to_dict()[0]
h = {}
for key in H:
    h[H[key]] = key
reduced_CT = pickle.load(open('./datasets/IBL/reduced_CT.pck', 'rb'))
cortical_regions = list(reduced_CT.keys())


def filter(region):
    trials = reduced_CT[region]
    keys = list(trials[0].keys())
    n = np.sum([c[keys[0]].shape[1] for c in trials])
    return n >= 50


def SD(data, null):
    return np.nanmean(np.asarray(data))
    # return np.nanmean(np.asarray(data) > (np.nanmean(null) + 3*np.nanstd(null)))
    # return np.nanmean(np.asarray(data) > np.nanmax(null))


# ------ Fig 6 for different Ls ------

import SD_squeeze

filename = f'./datasets/IBL/allperfs_L_{parhash(IBL_params)}.pck'
if os.path.exists(filename):
    [allperfs, allperfs_null] = pickle.load(open(filename, 'rb'))
else:
    allperfs = {}
    allperfs_null = {}
    for region in cortical_regions:
        if filter(region):
            p, pn = SD_squeeze.run(region)
            allperfs[region] = p
            allperfs_null[region] = pn
    pickle.dump([allperfs, allperfs_null], open(f'./datasets/IBL/allperfs_L_{parhash(IBL_params)}.pck', 'wb'))


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
    axs[i].set_ylim([0.45, 1.05])
    axs[i].grid()
    axs[i].axhline([np.nanmean(results['Separability'])], linewidth=2.0, color='b', linestyle='--')
plt.suptitle('mean perf')
f.savefig(f'./plots/revisions/mean_DP_squeeze_N={IBL_params["neurons"]}.pdf')


f, axs = plt.subplots(4, 5, figsize=(18, 12))
i = 0
best_L = {}

for region in allperfs:
    if filter(region):
        ax = axs.ravel()[i]
        ax.set_title(region)
        ax = axs.ravel()[i]
        Ls = list(allperfs[region].keys())[1:]
        SD_orig = SD(allperfs[region][-1], allperfs_null[region][-1])
        SD_squeezed = [SD(allperfs[region][L], allperfs_null[region][L]) for L in Ls]
        best_L[region], p = find_poly_maximum(Ls, SD_squeezed, degree=3)
        ax.plot(Ls, SD_squeezed, color='k')
        xs = np.linspace(1, 14, 100)
        ax.plot(xs, p(xs), color='k', linestyle='-', alpha=0.5)
        for l, L in enumerate(Ls):
            ax.plot(L, SD_squeezed[l], color=pltcolors[l + 1], marker='o')
        ax.axhline(SD_orig, color=pltcolors[0], linestyle='--', linewidth=2.0)
        ax.set_ylabel('SD threshold')
        ax.set_xticks(Ls)
        ax.set_ylim([0., 1.0])
        ax.set_xlim([0, 15])
        ax.set_xlabel('Latent dimensionality $L$')
        ax.axvline(np.log2(len(Ls)), color='g', linestyle='--')
        ax.axvline(best_L[region], color='r', linestyle='-')
        i += 1

f.savefig('./plots/revisions/SD_squeeze_all.pdf')


results = {
    'region': [],
    'Functional Dimensionality': [],
    'IC': [],
    'Teff': [],
    'PR': [],
    'Position in the hierarchy': []
}

for i, region in enumerate(list(reduced_CT.keys())):
    if filter(region):
        results['region'].append(str(region))
        print(region)
        results['Functional Dimensionality'].append(best_L[region])
        trials = reduced_CT[region]
        IC = len(trials[0].keys())
        results['IC'].append(IC)
        min_Ts = [np.sum([s[key].shape[0] for key in s]) for s in trials]
        results['Teff'].append(np.nanmean(min_Ts))
        results['PR'].append(participation_ratio(CT_to_X(trials, zscore=True)))
        results['Position in the hierarchy'].append(h[region])

f, ax = key_scatter(results, 'IC', 'Functional Dimensionality', groups, corr='reg', colors=100 * [pltcolors[0]], linecolor=pltcolors[0])
plt.plot(np.linspace(5, 16, 100), np.log2(np.linspace(5, 16, 100)), '--k')
f.savefig('./plots/revisions/IC_v_FD.pdf')

f, ax = key_scatter(results, 'PR', 'Functional Dimensionality', groups, corr='reg', colors=100 * [pltcolors[0]], linecolor=pltcolors[0])
f.savefig('./plots/revisions/PR_v_FD.pdf')

f, ax = key_scatter(results, 'Position in the hierarchy', 'Functional Dimensionality', groups, corr='reg', colors=100 * [pltcolors[0]],
            linecolor=pltcolors[0])
f.savefig('./plots/revisions/H_v_FD.pdf')

key_scatter(results, 'Teff', 'Functional Dimensionality', groups, corr='p', colors=100 * [pltcolors[0]], linecolor=pltcolors[0])
f.savefig('./plots/revisions/Teff_v_FD.pdf')

# PR vs FD
f, ax = plt.subplots(figsize=(4, 4))
ax.scatter(results['PR'], results['Functional Dimensionality'], alpha=0.7, edgecolors='k')
for i, region in enumerate(results['region']):
    ax.text(results['PR'][i]+0.05, results['Functional Dimensionality'][i]+0.05, region, fontsize=9)
ax.plot([2, 9], [2, 9], linestyle='--', color='k')
ax.set_xlabel('PR')
ax.set_ylabel('FD')
f.savefig('./plots/revisions/PR_v_FD_withdiagonal.pdf')

# log2 IC vs. FD
f, ax = plt.subplots(figsize=(4, 4))
ax.scatter(np.log2(results['IC']), results['Functional Dimensionality'], alpha=0.7, edgecolors='k')
for i, region in enumerate(results['region']):
    ax.text(np.log2(results['IC'])[i]+0.05, results['Functional Dimensionality'][i]+0.05, region, fontsize=9)
ax.plot([2, 9], [2, 9], linestyle='--', color='k')
ax.set_xlabel('log2(IC)')
ax.set_ylabel('FD')
f.savefig('./plots/revisions/logIC_v_FD_withdiagonal.pdf')


# ------ SDvsN plotting ------

import matplotlib
matplotlib.use('MacOSX')  # Try 'TkAgg' first, as it’s more likely pre-installed on macOS
import matplotlib.pyplot as plt
plt.ion()

import SD_vs_N
Ns = SD_vs_N.Ns
region = 'MOp'
mps = {}
aucs = {}
seps = {}

for L in [None, 1, 2, 3, 4, 5]:
    mp, auc, sep = SD_vs_N.run(region, L)
    mps[L] = mp
    aucs[L] = auc
    seps[L] = sep

f, axs = plt.subplots(1, 3, figsize=(10, 3.5))
axs[0].set_xlabel('N')
axs[1].set_xlabel('N')
axs[2].set_xlabel('N')
axs[0].set_ylabel('Mean DP')
axs[1].set_ylabel('1-AUC')
axs[2].set_ylabel('Separability')

for L in [None, 1, 2, 3, 4, 5]:
    if L is not None:
        axs[0].plot(Ns, mps[L], label=f'L={L}', marker='o')
        axs[1].plot(Ns, 1-aucs[L], label=f'L={L}', marker='o')
        axs[2].plot(Ns, seps[L], label=f'L={L}', marker='o')
    else:
        axs[0].plot(Ns, mps[L], label=f'original', marker='o')
        axs[1].plot(Ns, 1-aucs[L], label=f'original', marker='o')
        axs[2].plot(Ns, seps[L], label=f'original', marker='o')

axs[0].legend(fontsize=8)
axs[0].set_ylim([0.5, 0.85])
axs[0].set_xscale('log')
axs[1].set_xscale('log')
axs[2].set_xscale('log')
plt.suptitle('MOp')
f.savefig('./plots/revisions/MOp_SD_v_N.pdf')
