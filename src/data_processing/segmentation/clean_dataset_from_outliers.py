

from copy import deepcopy
from pathlib import Path
import random
from typing import List
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d
from tqdm import tqdm
from dataset.dataset_interface import DatasetInterface
from dataset.data_loading import BasicDataset
from utils.general_utils import split
from utils.transformation_utils import combine_point_clouds, imgs_to_pcd, pcd_to_imgs, resize, rs_ci

def _filter_clusters(start_cluster: o3d.geometry.PointCloud, clusters: List[o3d.geometry.PointCloud], max_distance: float):
    rejected_clusters = []
    accepted_clusters = []
    # Reject some clusters based on cluster size (min/max)
    #  for cluster in clusters:
    #      if len(cluster.points) > self.heuristics_max_cluster_size:
    #          print(f"Rejecting cluster with {len(cluster.points)} points (too many points)")
    #          rejected_clusters.append(cluster)
    #      elif len(cluster.points) < self.heuristics_min_cluster_size:
    #          print(f"Rejecting cluster with {len(cluster.points)} points (too few points)")
    #          rejected_clusters.append(cluster)
    _filter_clusters_recursive(start_cluster, clusters, accepted_clusters, rejected_clusters, max_distance)
    # if True: #  self.debug:
    #     vis_clusters = []
    #     for cluster in accepted_clusters:
    #         vis_cluster = deepcopy(cluster)
    #         vis_cluster.paint_uniform_color(np.array([0, 1, 0]))
    #         vis_clusters.append(vis_cluster)
    #     for cluster in rejected_clusters:
    #         vis_cluster = deepcopy(cluster)
    #         vis_cluster.paint_uniform_color(np.array([1, 0, 0]))
    #         vis_clusters.append(vis_cluster)
    #     o3d.visualization.draw_geometries(vis_clusters, "Data Processing: Cluster-based segmentation result")
    return accepted_clusters, rejected_clusters

def _filter_clusters_recursive(cluster, all_clusters, accepted_clusters, rejected_clusters, max_distance: float):
    possible_neighbours = [
        cluster 
        for cluster in all_clusters 
        if cluster not in accepted_clusters and cluster not in rejected_clusters
    ]
    neighbors, _ = _get_neighbors(cluster, possible_neighbours, max_distance)
    for neighbor in neighbors:
        if neighbor not in accepted_clusters and neighbor not in rejected_clusters:
            accepted_clusters.append(neighbor)
            _filter_clusters_recursive(neighbor, all_clusters, accepted_clusters, rejected_clusters, max_distance)
        #if neighbor not in accepted_clusters and neighbor not in rejected_clusters:
        #    neighbor_accepted, _, _ = self.color_filter.filter_clusters([neighbor])
        #    if len(neighbor_accepted) > 0:
        #        accepted_clusters.append(neighbor)
        #        self._filter_clusters_recursive(neighbor, all_clusters, accepted_clusters, rejected_clusters)
        #    else:
        #        rejected_clusters.append(neighbor)


def _get_neighbors(pcd: o3d.geometry.PointCloud, pcds: List[o3d.geometry.PointCloud], max_distance: float):
    neighbors = []
    not_neighbors = []
    for other in pcds:
        dist = np.min(np.asarray(pcd.compute_point_cloud_distance(other)))
        if dist < max_distance: # distance in meters
            neighbors.append(other)
        else:
            not_neighbors.append(other)
    return neighbors, not_neighbors


def select_largest_cluster(pcd: o3d.geometry.PointCloud, eps: float = 0.01, min_points=10, max_distance: float = 0.05, debug: bool = False):
    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    labels_vals, labels_counts = np.unique(labels, return_counts=True)

    if debug:
        max_label = labels.max()
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        rs_pcd_vis = deepcopy(pcd)
        rs_pcd_vis.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([rs_pcd_vis])

    # delete outlier cluster
    labels_vals = np.delete(labels_vals, 0)
    labels_counts = np.delete(labels_counts, 0)

    clusters = [
        pcd.select_by_index((labels == label).nonzero()[0])
        for label in labels_vals
    ]

    largest_cluster = clusters[np.argmax(labels_counts)]
    acc, _ = _filter_clusters(largest_cluster, clusters, max_distance)

    return combine_point_clouds(acc)

def clean_files(files, debug, in_dir, out_dir):
    for file in tqdm(files):
        raw_rs_rgb, raw_rs_depth, raw_zv_rgb, raw_zv_depth, mask = DatasetInterface.load(file)
        set = BasicDataset.preprocess_set(raw_rs_rgb, raw_rs_depth, mask, raw_zv_depth, BasicDataset.Config(
            scale=1,
            add_nan_mask_to_input=False,
            add_region_mask_to_input=False,
            normalize_depths=False,
            normalize_depths_min=0,
            normalize_depths_max=0,
            resize_region_to_fill_input=False
        ))

        rs_rgb = (set['image'][:3].numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        rs_depth = set['image'][3].numpy()
        zv_depth = set['label'].numpy().squeeze()

        raw_rs_pcd = imgs_to_pcd(raw_rs_rgb, raw_rs_depth.astype(np.float32), rs_ci)
        raw_zv_pcd = imgs_to_pcd(raw_zv_rgb, raw_zv_depth, rs_ci)
        rs_pcd = imgs_to_pcd(rs_rgb, rs_depth, rs_ci)
        zv_pcd = imgs_to_pcd(raw_zv_rgb, zv_depth, rs_ci)

        rs_pcd_clusterd = select_largest_cluster(rs_pcd, debug=debug)
        zv_pcd_clusterd = select_largest_cluster(zv_pcd, debug=debug)

        if debug:
            o3d.visualization.draw_geometries([raw_rs_pcd])
            o3d.visualization.draw_geometries([rs_pcd])
            o3d.visualization.draw_geometries([rs_pcd_clusterd])
            o3d.visualization.draw_geometries([raw_zv_pcd])
            o3d.visualization.draw_geometries([zv_pcd])
            o3d.visualization.draw_geometries([zv_pcd_clusterd])

        rs_rgb, rs_depth, rs_ul, rs_lr = pcd_to_imgs(rs_pcd_clusterd, rs_ci)
        zv_rgb, zv_depth, zv_ul, zv_lr = pcd_to_imgs(zv_pcd_clusterd, rs_ci)

        rs_rgb, rs_depth = resize(rs_rgb, rs_depth, rs_ul, rs_lr, cropped=False, resulting_shape=raw_rs_rgb.shape[:2])
        zv_rgb, zv_depth = resize(zv_rgb, zv_depth, zv_ul, zv_lr, cropped=False, resulting_shape=raw_zv_rgb.shape[:2])

        # update mask
        mask = np.where(np.logical_or(np.isnan(rs_depth), np.isnan(zv_depth))[..., None], False, mask)

        _, ax = plt.subplots(1, 3)
        ax[0].imshow(raw_rs_rgb)
        ax[1].imshow(rs_rgb)
        ax[2].imshow(np.sum(mask, axis=-1) > 0)
        plt.show()

        out_path = out_dir / file.relative_to(in_dir)
        DatasetInterface.save(rs_rgb, rs_depth, zv_rgb, zv_depth, mask, out_path)


def main():
    in_dir = Path("resources/images/calibrated_masked/not-cropped/ycb_video")
    out_dir = Path("resources/images/calibrated_masked_cleaned/not-cropped/ycb_video")
    files = DatasetInterface.get_paths_in_dir(in_dir)
    random.shuffle(files)
    debug = True
    jobs = 1 # cpu_count()

    files_chunked = split(files, jobs)

    Parallel(jobs)(
        delayed(clean_files)(files_chunk, debug, in_dir, out_dir)
        for files_chunk in files_chunked
    )


if __name__ == "__main__":
    main()