import numpy as np
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.ion()
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit
f, ax = plt.subplots(figsize=(4, 3))

# Parameters
noise_std = 0.01
C_values = np.linspace(0.01, 1.0, 20)
n_per_class = 1000
# Generate XOR data with noise

variables = {
    'semantic 1': np.array([0] * n_per_class + [0] * n_per_class + [1] * n_per_class + [1] * n_per_class),
    'semantic 2': np.array([0] * n_per_class + [1] * n_per_class + [0] * n_per_class + [1] * n_per_class),
    'XOR': np.array([0] * n_per_class + [1] * n_per_class + [1] * n_per_class + [0] * n_per_class)
}

for var in variables:
    X = np.vstack([
        np.random.normal(loc=[-1, -1], scale=noise_std, size=(n_per_class, 2)),
        np.random.normal(loc=[-1, 1], scale=noise_std, size=(n_per_class, 2)),
        np.random.normal(loc=[1, -1], scale=noise_std, size=(n_per_class, 2)),
        np.random.normal(loc=[1, 1], scale=noise_std, size=(n_per_class, 2)),
    ])
    y = variables[var]

    # Mesh grid for decision boundary visualization
    xx, yy = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-2, 2, 300))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    perfs = []

    # Plotting
    # fig, axes = plt.subplots(1, len(C_values), figsize=(20, 4), sharey=True)
    for C in C_values:
        perf = []
        for k in range(100):
            # Cross-validation splitter
            cv = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
            train_idx, test_idx = next(cv.split(X, y))
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            clf = LinearSVC(loss="hinge", C=C, max_iter=1000)
            clf.fit(X_train, y_train)
            perf_k = clf.score(X_test, y_test)
            perf.append(perf_k)

        perfs.append(np.nanmean(perf))
        # Decision function on the test grid
        # Z = clf.decision_function(grid_points).reshape(xx.shape)

        # Contours and test data
        # ax.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=["--", "-", "--"], colors="k")
        # ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, s=40, alpha=0.3)
        # ax.set_title(f"C = {C:.2f}\nDP={perf:.2f}")
        # ax.set_xlabel("x1")
        # ax.set_aspect("equal")

    # axes[0].set_ylabel("x2")
    # plt.suptitle("Linear SVM on Noisy XOR (Test Set) for Varying C", fontsize=16)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.show()

    ax.plot(C_values, perfs, '-', label=var, linewidth=4, alpha=0.5)

ax.set_ylim([0.45, 1.05])
ax.axhline([0.5], color='k', linewidth=2, alpha=0.5, linestyle='--')
ax.set_xlabel('C')
ax.set_ylabel('CV Decoding Performance')
ax.set_title('$\sigma=%.2f$' % noise_std)
plt.tight_layout()
plt.legend()
