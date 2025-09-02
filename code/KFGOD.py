import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from GB_generation_with_idx import get_GB


def gaussian_matrix(Data, r):
    n = Data.shape[0]
    m = Data.shape[1]
    transdata = np.zeros((n, m))
    transdata[:, 0:m] = Data
    temp = pdist(transdata, 'euclidean')
    temp = squareform(temp)
    temp = np.exp(-(temp ** 2) / r)
    return temp


def KFGOD(data, delta):
    n, m = data.shape
    LA = np.arange(0, m)
    weight1 = np.zeros((n, m))
    weight2 = np.zeros((n, m))

    Acc_A_a = np.zeros((n, m))
    for l in range(0, m):
        lA_d = np.setdiff1d(LA, l)

        NbrSet_tem = gaussian_matrix((np.matrix(data[:, l])).T, delta)
        NbrSet_temp, ia, ic = np.unique(NbrSet_tem, return_index=True, return_inverse=True, axis=0)

        for i in range(0, NbrSet_temp.shape[0]):
            i_tem = np.where(ic == i)[0]
            data_tem = data[:, lA_d]
            NbrSet_tmp = gaussian_matrix(data_tem, delta)
            a = 1 - NbrSet_tmp
            b = np.tile(NbrSet_temp[i, :], (n, 1))
            Low_A = sum((np.minimum(a + b - np.multiply(a, b) + np.multiply(np.sqrt(2 * a - np.multiply(a, a)),
                                                                            np.sqrt(2 * b - np.multiply(b, b))),
                                    1)).min(-1))

            a = NbrSet_tmp
            Upp_A = sum((np.maximum(
                np.multiply(a, b) - np.multiply(np.sqrt(1 - np.multiply(a, a)), np.sqrt(1 - np.multiply(b, b))),
                0)).max(-1))

            Acc_A_a[i_tem, l] = Low_A / Upp_A
            weight2[i_tem, l] = 1 - (sum(NbrSet_temp[i, :]) / n) ** (1 / 3)
            weight1[i_tem, l] = (sum(NbrSet_temp[i, :]) / n)

    GOD = np.zeros((n, m))
    for col in range(m):
        GOD[:, col] = 1 - (Acc_A_a[:, col]) * weight1[:, col]

    OD_gb = np.array(np.mean(GOD * weight2, axis=1))
    return OD_gb


if __name__ == "__main__":
    data = pd.read_csv("./Example.csv").values
    X = data[:, :-1]
    n, m = X.shape
    labels = data[:, -1]
    ID = (X >= 1).all(axis=0) & (X.max(axis=0) != X.min(axis=0))
    scaler = MinMaxScaler()
    if any(ID):
        scaler = MinMaxScaler()
        X[:, ID] = scaler.fit_transform(X[:, ID])

    GBs = get_GB(X)
    n_gb = len(GBs)
    print(f"The number of Granular-ball: {n_gb}")
    
    centers = np.zeros((n_gb, m))
    for idx, gb in enumerate(GBs):
        centers[idx] = np.mean(gb[:,:-1], axis=0)
        
    delta = 0.3
    OD_gb = KFGOD(centers, delta)
    
    '''Map to samples'''
    OD = np.zeros(n)
    for idx, gb in enumerate(GBs):
        point_idxs = gb[:,-1].astype('int')
        OD[point_idxs] = OD_gb[idx]
    print(OD_gb)
