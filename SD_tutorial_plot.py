from source import *
import matplotlib
import numpy as np

matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

plt.ion()
from SD_tutorial import run

# Varying Alpha
N = 640
T = 40
nreps = 100
iterations = 10

alphas = np.linspace(0, 3.0, 11)
Rs = [1.0, 2.0, 3.0, 4.0, 5.0]
sigmas = [0.5, 1, 2, 3, 4]

PR_all = {}
SD_all = {}
SEP_all = {}

PR_all_std = {}
SD_all_std = {}
SEP_all_std = {}

for r in Rs:
    for sigma in sigmas:
        alphas = np.linspace(0, 3.0, 11)
        PRs = np.zeros((len(alphas), nreps))
        SDs = np.zeros((len(alphas), nreps))
        SEPs = np.zeros((len(alphas), nreps))

        for i, alpha in enumerate(alphas):
            PRs[i], SEPs[i], SDs[i] = run(alpha, r, sigma, N, T, nreps, iterations)

        f, ax = plt.subplots(figsize=(4, 3))
        ax.errorbar(alphas, np.nanmean(SDs, 1), 2 * np.nanstd(SDs, 1), marker='s', label='SD', capsize=4, alpha=0.8)
        ax.set_ylabel('SD', color=pltcolors[0])
        ax2 = ax.twinx()
        ax2.errorbar(alphas, np.nanmean(PRs, 1), 2 * np.nanstd(PRs, 1), marker='o', linestyle='--',
                     label='PR', capsize=4, color='b', alpha=0.8)
        ax2.set_ylabel('PR', color='b')

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

        ax.set_xlabel('Distortion $\\alpha$')
        ax.set_title('$\sigma=%.2f$, $R=%.2f$' % (sigma, r), fontsize=10)

        f.savefig('./plots/IBL/cuboid/r=%.2f_N=%u_T=%u_s=%.2f.pdf' % (r, N, T, sigma))
        plt.close(f)
        SD_all[(sigma, r)] = np.nanmean(SDs, 1)
        SD_all_std[(sigma, r)] = np.nanstd(SDs, 1)

        PR_all[(sigma, r)] = np.nanmean(PRs, 1)
        PR_all_std[(sigma, r)] = np.nanstd(PRs, 1)

        SEP_all[(sigma, r)] = np.nanmean(SEPs, 1)
        SEP_all_std[(sigma, r)] = np.nanstd(SEPs, 1)

    f, axs = plt.subplots(1, 2, figsize=(8, 4))
    for sigma in sigmas:
        axs[0].errorbar(x=PR_all[(sigma, r)], y=SD_all[(sigma, r)], yerr=SD_all_std[(sigma, r)],
                        xerr=PR_all_std[(sigma, r)], marker='o', alpha=0.666, label='$\sigma=%.1f$' % sigma)
        axs[1].errorbar(x=PR_all[(sigma, r)], y=SEP_all[(sigma, r)], yerr=SEP_all_std[(sigma, r)],
                        xerr=PR_all_std[(sigma, r)], marker='o', alpha=0.666, label='$\sigma=%.1f$' % sigma)
    plt.legend(fontsize=8)
    axs[0].set_xlabel('PCA Dimensionality (PR)')
    axs[0].set_ylabel('Separability')
    axs[0].set_ylim([0, 1.05])
    axs[0].set_xlim([1, 15])

    axs[1].set_xlabel('PCA Dimensionality (PR)')
    axs[1].set_ylabel('Decodability')
    axs[1].set_ylim([0.45, 1.05])
    linenull(axs[1])
    axs[1].set_xlim([1, 15])

    plt.suptitle('L=4 variables (M$_{IC}$=16), $\gamma$=%.1f' % r, fontsize=11)
    f.savefig('./plots/IBL/cuboid/R=%.1f.pdf' % r)

# rearrange data so that R varies
PR_r = {}
PR_r_std = {}
SD_r = {}
SD_r_std = {}

alphas = np.linspace(0, 3.0, 11)
idx = [0, 3, 7, 10]

for sigma in sigmas:
    for i in idx:
        PR_r[(sigma, alphas[i])] = [PR_c_all[(sigma, r)][i] for r in Rs]
        PR_r_std[(sigma, alphas[i])] = [PR_c_all_std[(sigma, r)][i] for r in Rs]
        SD_r[(sigma, alphas[i])] = [SD_all[(sigma, r)][i] for r in Rs]
        SD_r_std[(sigma, alphas[i])] = [SD_all_std[(sigma, r)][i] for r in Rs]

for alpha in alphas[idx]:
    f, ax = plt.subplots(figsize=(4, 4))
    for sigma in sigmas:
        ax.errorbar(x=PR_r[(sigma, alpha)], y=SD_r[(sigma, alpha)], yerr=SD_r_std[(sigma, alpha)],
                    xerr=PR_r_std[(sigma, alpha)], marker='o', alpha=0.666, label='$\sigma=%.1f$' % sigma)
    plt.legend(fontsize=8)
    ax.set_xlabel('PCA Dimensionality (PR)')
    ax.set_ylabel('Shattering Dimensionality')
    ax.set_ylim([0, 1.05])
    ax.set_xlim([1, 15])
    ax.set_title('L=4 variables (M$_{IC}$=16), $\\alpha$=%.1f' % alpha, fontsize=11)
    f.savefig('./plots/IBL_synthetic/a=%.1f.pdf' % alpha)
