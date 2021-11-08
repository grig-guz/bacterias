
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def dist_util(src_pos, dst_pos, criterion):
    src_dst_dists = euclidean_distances(src_pos, dst_pos)
    if type(criterion) == float:
        min_dists_idx = np.argmin(src_dst_dists, axis=1)
        to_remain = src_dst_dists[np.arange(len(src_pos)), min_dists_idx] > criterion
    else:
        min_dists_idx = np.argsort(src_dst_dists, axis=1)
        to_remain = min_dists_idx[:, :criterion]
    return to_remain, min_dists_idx


