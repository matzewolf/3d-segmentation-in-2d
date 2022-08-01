import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans


def jitter_point_cloud(points_3d, sigma = 0.01,
                       clip= 0.05):
    """
    Randomly jitters each point independently.

    :param points_3d: BxNx3 array, original batch of point clouds
    :param sigma: Variance of the jitter
    :param clip: maximum absolute value of the jitter
    :return: BxNx3 array, jittered batch of point clouds
    """
    assert clip > 0
    return points_3d + np.clip(sigma * np.random.randn(*points_3d.shape),
                               -clip, clip)


def gpgl2_segmentation(
        points_3d,
        num_cuts = 32,
        size_sub = 16,
        size_top = 16
):
    k_means = KMeans(n_clusters=num_cuts, n_init=1, max_iter=100)
    points_3d = points_3d + np.random.rand(*points_3d.shape) * 1e-6
    num_points = points_3d.shape[0]
    dist_mat = k_means.fit_transform(points_3d)

    node_top, labels = graph_cut(points_3d, dist_mat, num_points, num_cuts)

    aij_mat = squareform(pdist(node_top), checks=False)
    H = nx.from_numpy_matrix(aij_mat)
    pos_spring = nx.spring_layout(H)
    pos_spring = np.array([pos for idx, pos in sorted(pos_spring.items())])
    pos_top = gpgl_layout_push(pos_spring, size_top)

    pos_cuts = []
    for i_cut in range(num_cuts):
        pos_cut_3D = points_3d[labels == i_cut, :]

        if len(pos_cut_3D) < 5:
            pos_raw = [[0, 0], [0, 1], [1, 1], [1, 0]]
            pos = pos_raw[:len(pos_cut_3D)]
            pos_cuts.append(pos)
            continue

        aij_mat = squareform(pdist(pos_cut_3D), checks=False)
        H = nx.from_numpy_matrix(aij_mat)
        pos_spring = nx.spring_layout(H)
        pos_spring = np.array([pos for idx, pos in sorted(pos_spring.items())])
        pos = gpgl_layout_push(pos_spring, size_sub)

        pos_cuts.append(pos)

    # combine all layout positions
    cuts_count = np.zeros(num_cuts).astype(np.int64)
    pos_all = []
    for idx in range(num_points):
        label = labels[idx]
        pos_all.append(
            pos_cuts[label][cuts_count[label]] + pos_top[label] * size_sub)
        cuts_count[label] += 1
    pos_all = np.array(pos_all)

    num_nodes_m = len(np.unique(pos_all, axis=0))
    node_loss_rate = (1 - num_nodes_m / num_points)
    return pos_all, node_loss_rate


def graph_cut(data, dist_mat,
              num_points, num_cuts):
    num_cutpoints = int(num_points / num_cuts)
    cutpoints_threshold = np.ceil(num_cutpoints * 1.2)
    cluster = np.argmin(dist_mat, axis=-1)
    mask = np.zeros([num_points, num_cuts])
    for m, c in zip(mask, cluster):
        m[c] = 1
    flow_mat = np.zeros([num_cuts, num_cuts])

    # separate point cloud into num_cuts clusters
    for i_loop in range(500):
        loss_mask = mask.sum(0)
        order_list = np.argsort(loss_mask)
        if loss_mask.max() <= cutpoints_threshold + 1:
            break
        for i_order, order in zip(range(len(order_list)), order_list):
            if loss_mask[order] > cutpoints_threshold:
                idxs = np.where(mask[:, order])[0]
                idys_ori = order_list[:i_order]
                idys = []
                for idy in idys_ori:
                    if flow_mat[order, idy] >= 0:
                        idys.append(idy)

                mat_new = dist_mat[idxs, :]
                mat_new = mat_new[:, idys]
                cost_list_row = mat_new.argmin(-1)
                cost_list_col = mat_new.min(-1)

                row = cost_list_col.argmin(-1)
                col = cost_list_row[row]

                target_idx = [idxs[row], idys[col]]
                mask[target_idx[0], order] = 0
                mask[target_idx[0], target_idx[1]] = 1
                flow_mat[order, target_idx[1]] = 1
                flow_mat[target_idx[1], order] = -1

    center_pos = []
    for i_cut in range(num_cuts):
        if mask[:, i_cut].sum() > 0:
            center_pos.append(data[mask[:, i_cut].astype(np.bool), :].mean(0))
        else:
            center_pos.append([0, 0])
    labels = mask.argmax(-1)
    return np.array(center_pos), labels


def gpgl_layout_push(pos, size):
    dist_mat = pdist(pos)
    scale1 = 1 / dist_mat.min()
    scale2 = (size-2) / (pos.max() - pos.min())
    scale = np.min([scale1, scale2])
    pos *= scale

    pos_quat = np.round(pos).astype(np.int)
    pos_quat = pos_quat - np.min(pos_quat, axis=0) + np.array([1, 1])
    pos_unique, count = np.unique(pos_quat, axis=0, return_counts=True)
    # node_loss = np.sum(count)-len(count)
    # print('node_loss',node_loss)

    mask = np.zeros((size, size)).astype(np.int)
    for pt in pos_quat:
        mask[pt[0], pt[1]] += 1

    for i_loop in range(50):
        if mask.max() <= 1:
            # print("early stop")
            break
        idxs = np.where(count > 1)[0]
        for idx in idxs:
            pos_overlap = pos_unique[idx]
            dist = cdist(pos_quat, [pos_overlap])
            idy = np.argmin(dist)

            b_down = np.maximum(pos_overlap[0] - 1, 0)
            b_up = np.minimum(pos_overlap[0] + 2, size)
            b_left = np.maximum(pos_overlap[1] - 1, 0)
            b_right = np.minimum(pos_overlap[1] + 2, size)

            mask_target = mask[b_down:b_up, b_left:b_right]
            if mask_target.min() == 0:
                pos_target = np.unravel_index(np.argmin(mask_target),
                                              mask_target.shape)
                pos_mask = pos_target + np.array([b_down, b_left])
            else:
                pos_empty = np.array(np.where(mask == 0)).T
                dist = cdist(pos_empty, [pos_overlap])
                pos_target = pos_empty[np.argmin(dist)]
                direction = pos_target - pos_overlap
                direction1 = np.round(direction / np.linalg.norm(direction))
                pos_mask = pos_overlap + direction1.astype(np.int)

            pos_quat[idy] = pos_mask
            mask[pos_overlap[0], pos_overlap[1]] -= 1
            mask[pos_mask[0], pos_mask[1]] += 1
            pos_unique, count = np.unique(pos_quat, axis=0, return_counts=True)
    return pos_quat
