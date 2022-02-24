

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
import random
from typing import List
from joblib import Parallel, delayed, cpu_count
from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d
from tqdm import tqdm
from dataset.dataset_interface import DatasetInterface
from utils.general_utils import split
from utils.transformation_utils import combine_point_clouds, imgs_to_pcd, pcd_to_imgs, resize, rs_ci


class DBClustering():

    def __init__(self,
                 heuristics_min_cluster_size: int = 0,
                 heuristics_max_cluster_size: int = np.inf,
                 min_points: int = 10,
                 eps: float = 0.02,
                 max_neigbour_distance=0.05,
                 debug: bool = False
                 ):
        self.heuristics_min_cluster_size = heuristics_min_cluster_size
        self.heuristics_max_cluster_size = heuristics_max_cluster_size
        self.min_points = min_points
        self.eps = eps
        self.max_neigbour_distance = max_neigbour_distance
        self.debug = debug

    def _filter_clusters(self, start_cluster: o3d.geometry.PointCloud, clusters: List[o3d.geometry.PointCloud]):
        rejected_clusters = []
        accepted_clusters = []
        # Reject some clusters based on cluster size (min/max)
        for cluster in clusters:
            if len(cluster.points) > self.heuristics_max_cluster_size:
                print(f"Rejecting cluster with {len(cluster.points)} points (too many points)")
                rejected_clusters.append(cluster)
            elif len(cluster.points) < self.heuristics_min_cluster_size:
                print(f"Rejecting cluster with {len(cluster.points)} points (too few points)")
                rejected_clusters.append(cluster)
        self._filter_clusters_recursive(start_cluster, clusters, accepted_clusters, rejected_clusters)
        return accepted_clusters, rejected_clusters

    def _filter_clusters_recursive(self, cluster, all_clusters, accepted_clusters, rejected_clusters):
        possible_neighbours = [
            cluster
            for cluster in all_clusters
            if cluster not in accepted_clusters and cluster not in rejected_clusters
        ]
        neighbors, _ = self._get_neighbors(cluster, possible_neighbours)
        for neighbor in neighbors:
            if neighbor not in accepted_clusters and neighbor not in rejected_clusters:
                accepted_clusters.append(neighbor)
                self._filter_clusters_recursive(neighbor, all_clusters, accepted_clusters,
                                                rejected_clusters)
            # if neighbor not in accepted_clusters and neighbor not in rejected_clusters:
            #    neighbor_accepted, _, _ = self.color_filter.filter_clusters([neighbor])
            #    if len(neighbor_accepted) > 0:
            #        accepted_clusters.append(neighbor)
            #        self._filter_clusters_recursive(neighbor, all_clusters, accepted_clusters, rejected_clusters)
            #    else:
            #        rejected_clusters.append(neighbor)

    def _get_neighbors(self, pcd: o3d.geometry.PointCloud, pcds: List[o3d.geometry.PointCloud]):
        neighbors = []
        not_neighbors = []
        for other in pcds:
            dist = np.min(np.asarray(pcd.compute_point_cloud_distance(other)))
            if dist < self.max_neigbour_distance:  # distance in meters
                neighbors.append(other)
            else:
                not_neighbors.append(other)
        return neighbors, not_neighbors

    def select_largest_cluster(self, pcd: o3d.geometry.PointCloud):
        labels = np.asarray(pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points, print_progress=False))
        labels_vals, labels_counts = np.unique(labels, return_counts=True)

        if self.debug:
            max_label = labels.max()
            colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0
            rs_pcd_vis = deepcopy(pcd)
            rs_pcd_vis.colors = o3d.utility.Vector3dVector(colors[:, :3])
            o3d.visualization.draw_geometries([rs_pcd_vis])

        # delete outlier cluster
        if -1 in labels_vals:
            labels_vals = np.delete(labels_vals, 0)
            labels_counts = np.delete(labels_counts, 0)

        clusters = [
            pcd.select_by_index((labels == label).nonzero()[0])
            for label in labels_vals
        ]

        largest_cluster = clusters[np.argmax(labels_counts)]
        acc, rej = self._filter_clusters(largest_cluster, clusters)

        if self.debug:
            vis_clusters = []
            for cluster in acc:
                vis_cluster = deepcopy(cluster)
                vis_cluster.paint_uniform_color(np.array([0, 1, 0]))
                vis_clusters.append(vis_cluster)
            for cluster in rej:
                vis_cluster = deepcopy(cluster)
                vis_cluster.paint_uniform_color(np.array([1, 0, 0]))
                vis_clusters.append(vis_cluster)
            o3d.visualization.draw_geometries(vis_clusters, "Data Processing: Cluster-based segmentation result")

        return combine_point_clouds(acc)


def clean_files(files, debug, db_clustering: DBClustering, in_dir, out_dir):
    for file in tqdm(files):
        raw_rs_rgb, raw_rs_depth, raw_zv_rgb, raw_zv_depth, mask = DatasetInterface.load(file)

        mask = mask.squeeze()

        # apply current mask
        rs_depth = np.where(mask, raw_rs_depth, np.nan)
        zv_depth = np.where(mask, raw_zv_depth, np.nan)

        # compute pcds
        rs_pcd = imgs_to_pcd(raw_rs_rgb, rs_depth, rs_ci)
        zv_pcd = imgs_to_pcd(raw_zv_rgb, zv_depth, rs_ci)

        if debug:
            o3d.visualization.draw_geometries([rs_pcd])

        # cluster rs pcd
        rs_pcd_clusterd = db_clustering.select_largest_cluster(rs_pcd)

        if debug:
            o3d.visualization.draw_geometries([rs_pcd_clusterd])
            o3d.visualization.draw_geometries([zv_pcd])

        # cluster zv pcd
        zv_pcd_clusterd = db_clustering.select_largest_cluster(zv_pcd)

        if debug:
            o3d.visualization.draw_geometries([zv_pcd_clusterd])

        rs_rgb, rs_depth, rs_ul, rs_lr = pcd_to_imgs(rs_pcd_clusterd, rs_ci)
        zv_rgb, zv_depth, zv_ul, zv_lr = pcd_to_imgs(zv_pcd_clusterd, rs_ci)

        rs_rgb, rs_depth = resize(rs_rgb, rs_depth, rs_ul, rs_lr, cropped=False, resulting_shape=raw_rs_rgb.shape[:2])
        zv_rgb, zv_depth = resize(zv_rgb, zv_depth, zv_ul, zv_lr, cropped=False, resulting_shape=raw_zv_rgb.shape[:2])

        # update mask
        mask = np.where(np.logical_or(np.isnan(rs_depth), np.isnan(zv_depth)), False, mask)

        _, ax = plt.subplots(1, 3)
        ax[0].imshow(raw_rs_rgb)
        ax[1].imshow(rs_rgb)
        ax[2].imshow(mask)
        plt.show()

        out_path = out_dir / file.relative_to(in_dir)
        DatasetInterface.save(rs_rgb, rs_depth, zv_rgb, zv_depth, mask, out_path)


def main(args):
    in_dir = args.in_dir
    out_dir = args.out_dir
    files = DatasetInterface.get_paths_in_dir(in_dir)
    # random.shuffle(files)
    debug = args.debug
    jobs = args.jobs

    files_chunked = split(files, jobs)

    db_clustering = DBClustering(
        heuristics_min_cluster_size=1000,
        max_neigbour_distance=0.15,
        debug=debug
    )

    Parallel(jobs)(
        delayed(clean_files)(files_chunk, debug, db_clustering, in_dir, out_dir)
        for files_chunk in files_chunked
    )


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("in_dir", type=Path)
    argparse.add_argument("out_dir", type=Path)
    argparse.add_argument("--jobs", type=int, default=cpu_count())
    argparse.add_argument("--debug", action="store_true")
    main(argparse.parse_args())
