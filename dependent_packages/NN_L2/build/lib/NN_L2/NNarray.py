from numpy import any, argsort, arange, delete, greater_equal, nonzero, ones
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree
try:
    import faiss
    use_faiss = True
except:
    print("faiss module is not installed, hence not used", flush=True)
    use_faiss = False


class MyNNBase:
    def __init__(self, X, **kwargs):
        self.n, self.d = X.shape

    def __call__(self, XOther, k):
        raise NotImplementedError


class MyNNFaiss(MyNNBase):
    def __init__(self, X, **kwargs):
        super(MyNNFaiss, self).__init__(X, **kwargs)
        if self.n < 1e5:
            self.index = faiss.IndexFlatL2(self.d)
        else:
            quantizer = faiss.IndexFlatL2(self.d)
            self.index = faiss.IndexIVFFlat(quantizer, self.d,
                                            min(self.n, 1024))
            self.index.train(self.n, X)
            self.index.nprobe = min(self.n, 256)
        self.index.add(self.n, X)

    def __call__(self, XOther, k):
        _, NN_sub = self.index.search(XOther, k)
        return NN_sub


class MyNNScipy(MyNNBase):
    def __init__(self, X, **kwargs):
        super(MyNNScipy, self).__init__(X, **kwargs)
        self.index = KDTree(X, leaf_size=128, metric='euclidean')

    def __call__(self, XOther, k):
        return self.index.query(XOther, k, return_distance=False)


def NN_L2(locs, m: int):
    n, d = locs.shape
    NN = -ones((n, m + 1), dtype=int)
    mult = 2
    maxVal = min(m * mult + 1, n)
    dist_mat = cdist(locs[:maxVal, :], locs[:maxVal, :])
    odrM = argsort(dist_mat)
    for i in range(maxVal):
        NN_row = odrM[i, :]
        NN_row = NN_row[NN_row <= i]
        NN_len = min(NN_row.shape[0], m + 1)
        NN[i, :NN_len] = NN_row[:NN_len]
    query_idx = arange(maxVal, n)
    m_search = m
    while query_idx.size > 0:
        max_idx = query_idx.max()
        m_search = min(max_idx + 1, 2 * m_search)
        if use_faiss:
            NN_wrap = MyNNFaiss(locs[: max_idx + 1, :])
        else:
            NN_wrap = MyNNScipy(locs[: max_idx + 1, :])
        NN_sub = NN_wrap(locs[query_idx, :], int(m_search))
        less_than_i = NN_sub <= query_idx[:, None]
        num_less_than_i = less_than_i.sum(1)
        idx_less_than_i = nonzero(greater_equal(num_less_than_i, m + 1))[0]
        for i in idx_less_than_i:
            NN[query_idx[i]] = NN_sub[i, less_than_i[i, :]][: m + 1]
            if NN[query_idx[i], 0] != query_idx[i]:
                try:
                    idx = nonzero(NN[query_idx[i]] == query_idx[i])[0][0]
                    NN[query_idx[i], idx] = NN[query_idx[i], 0]
                    NN[query_idx[i], 0] = query_idx[i]
                except IndexError as inst:
                    NN[query_idx[i], 0] = query_idx[i]
        query_idx = delete(query_idx, idx_less_than_i, 0)
    if any(NN[:, 0] != arange(n)):
        print(f"There are very close locations and "
              f"NN[:, 0] != np.arange(n)", flush=True)
    return NN
