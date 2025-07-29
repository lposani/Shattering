from source import *
import sys
from source.analysis.Dimensionality import *
from source.analysis.Clustering import *
from IBL_settings import *
mkdir('./plots/IBL/cuboid')
mkdir('./plots/IBL/cuboid/SD')
mkdir('./datasets/IBL/cuboid_cache')


def generate_latent_vectors(L):
    return np.asarray(list(itertools.product([0, 1], repeat=L))) * 2 - 1


def generate_latent_representations(L, alpha, sigma, T=100, N=400, P=None, weights=None):
    if P is None:
        v = generate_latent_vectors(L)
        P = 2 ** L
        labels = [''.join(x) for x in ((v + 1) / 2).astype(int).astype(str)]
    else:
        v = 2*np.random.rand(P, L) - 1  # uniform in [-1, 1] L-cube
        labels = [f'{i}' for i in range(P)]

    if weights is not None:
        v = v * weights

    U = np.random.randn(L, N)
    trials = {}

    for i in range(P):
        trials[labels[i]] = np.dot(v[i], U) + \
                            np.random.randn(N) * alpha * 2 * np.sqrt(L) + \
                            np.random.randn(T, N) * sigma * np.sqrt(L) * np.sqrt(2 + 2. * alpha ** 2.)
    return trials


def dimensionality_measures(alpha, r, sigma, N, T, L=4, nreps=100, iteration=0):
    weights = np.ones(L)
    weights[0] *= r
    np.random.seed(iteration)
    trials = generate_latent_representations(L=L, alpha=alpha, sigma=sigma, T=T, N=N, weights=weights)

    # Compute PR
    X = CT_to_X([trials])
    pr = participation_ratio(X)

    # Compute separability and SD
    cache_name = f'cuboid_a={alpha:.2f}_r={r:.2f}_N={N}_T={T}_s={sigma}_L={L}_nrep={nreps}_i={iteration}_{parhash(decoding_params)}'
    perfs, perfs_null, fingerprints = shattering_dimensionality(trials,
                                                                nreps=nreps,
                                                                nnulls=100,
                                                                n_neurons=None,
                                                                region=f'a={alpha:.2f}_r{r:.2f}_s{sigma}',
                                                                folder='cuboid',
                                                                IC=False,
                                                                cache_name=cache_name,
                                                                convert_dic=False,
                                                                **decoding_params)
    sd = np.nanmean(perfs)
    sep = np.nanmean(perfs > np.nanmax(perfs_null))

    return pr, sep, sd


def run(alpha, r, sigma, N, T, nreps=100, iterations=10):
    PRs = np.zeros(iterations)
    SDs = np.zeros(iterations)
    SEPs = np.zeros(iterations)

    for k in range(iterations):
        path = './datasets/IBL/cuboid_cache/res_r=%.2f_N=%u_T=%u_s=%.2f_nreps=%u_niter=%u.pck' \
               % (r, N, T, sigma, nreps, k)
        if os.path.exists(path):
            [pr, sep, sd] = pickle.load(open(path, 'rb'))
        else:
            pr, sep, sd = dimensionality_measures(alpha=alpha, r=r, sigma=sigma, N=N, T=T, iteration=k, nreps=nreps)
            pickle.dump([pr, sep, sd], open(path, 'wb'))

        PRs[k] = pr
        SDs[k] = sd
        SEPs[k] = sep

    return PRs, SEPs, SDs


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Please specify alpha and R")
        sys.exit(1)

    alpha = float(sys.argv[1])
    r = float(sys.argv[2])
    sigmas = [0.5, 1.0, 2.0, 3.0, 4.0]
    for sigma in sigmas:
        run(alpha=alpha, r=r, sigma=sigma, N=640, T=40, nreps=100, iterations=10)

