import numpy as np
import scipy.io as scio

def get_satellite(path='./satellite.mat', seed=2021):
    rdg = np.random.RandomState(seed)
    datamat = scio.loadmat(path)
    X = datamat['X']    # (7200,6)
    y = datamat['y']    # (7200,1)
    perm = rdg.permutation(len(X))

    X = X[perm[:len(X)]]
    y = y[perm[:len(y)]]
    y = y.astype(np.float)
    y[np.where(y == 0)] = -1

    Labeled_x = X[:500]
    Labeled_y = y[:500, 0]

    Unlabeled_x = X[500:]
    Unlabeled_y = y[500:, 0]

    return Labeled_x, Labeled_y, Unlabeled_x, Unlabeled_y

def get_data32(seed=0):
    rdg = np.random.RandomState(seed)

    dim = 32

    cov0_min, cov0_max = 1, 4
    mean0_min, mean0_max = 0, 3
    cov0 = (rdg.randint(cov0_min, cov0_max, (dim, dim)) * np.eye(dim, dim)).tolist()
    mean0 = rdg.randint(mean0_min, mean0_max, (dim)).tolist()
    x0 = rdg.multivariate_normal(mean0, cov0, size=700)

    cov1_min, cov1_max = 1, 4
    mean1_min, mean1_max = 0, 3
    cov1 = (rdg.randint(cov1_min, cov1_max, (dim, dim)) * np.eye(dim, dim)).tolist()
    mean1 = rdg.randint(mean1_min, mean1_max, (dim)).tolist()
    x1 = rdg.multivariate_normal(mean1, cov1, size=700)


    x0_L = x0[:50]
    x0_U = x0[50:]

    x1_L = x1[:50]
    x1_U = x1[50:]

    Lx = np.r_[x0_L, x1_L]
    Ly = np.r_[np.ones(len(x0_L)), -np.ones(len(x1_L))]

    Ux = np.r_[x0_U, x1_U]
    Uy = np.r_[np.ones(len(x0_U)), -np.ones(len(x1_U))]

    indices = rdg.permutation(len(Lx))
    Lx = Lx[indices]
    Ly = Ly[indices]

    indices = rdg.permutation(len(Ux))
    Ux = Ux[indices]
    Uy = Uy[indices]

    return Lx, Ly, Ux, Uy