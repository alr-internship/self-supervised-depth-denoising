from argparse import ArgumentParser
import random
from re import A
import cv2
from copy import deepcopy
from pathlib import Path
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d
import Regions as RG

from tqdm import tqdm
from data_processing.segmentation.clean_dataset_from_outliers import DBClustering
from dataset.dataset_interface import DatasetInterface
from utils.transformation_utils import imgs_to_pcd, pcd_to_imgs, resize, rs_ci, combine_point_clouds
from utils.visualization_utils import to_rgb
from utils.general_utils import split


def crop_by_bbox(pcd, bbox_file: Path, debug: bool):
    padding = [0.1, 0.1, 0.1]  # in meterc

    # o3d.visualization.draw_geometries([combined_pcd])
    if not bbox_file.exists():
        v = o3d.visualization.VisualizerWithEditing()
        v.create_window()
        v.add_geometry(pcd)
        v.run()
        v.destroy_window()
        picked_points_indices = v.get_picked_points()
        points = np.array(np.asarray(pcd.points)[picked_points_indices])
        # min_bound = np.min(points, axis=0) - padding
        # max_bound = np.max(points, axis=0) + padding
        # print(f"Bounding Box: {min_bound}:{max_bound}")

        points_o3d = o3d.utility.Vector3dVector(points)
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(points_o3d)
        bbox_padded = o3d.geometry.OrientedBoundingBox(
            center=bbox.center,
            R=bbox.R,
            extent=bbox.extent + padding
        )

        np.savez(bbox_file, center=bbox_padded.center, R=bbox_padded.R, extent=bbox_padded.extent)

    else:
        data = np.load(bbox_file)
        bbox_padded = o3d.geometry.OrientedBoundingBox(
            center=data['center'],
            R=data['R'],
            extent=data['extent']
        )

    cropped_pcd = pcd.crop(bbox_padded)

    if debug:
        o3d.visualization.draw_geometries([cropped_pcd])
    return cropped_pcd


def remove_clusters_by_region(pcd, points_file: Path, add_outliers: bool, debug: bool):
    # filter_objects(zv_pcd)
    RGKNN = RG.RegionGrowing()
    RGKNN.SetDataThresholds(pcd, t_a=7.5)
    RGKNN.RGKnn()
    labels = RGKNN.ReLabeles()
    # Visualizer
    max_label = len(RGKNN.Clusters)

    if debug:
        print(f"point cloud has {max_label} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 1] = 1         # set to white for small clusters (label - 0 )
        v_zv_pcd = deepcopy(pcd)
        v_zv_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([v_zv_pcd])

    clusters = np.asarray([
        pcd.select_by_index((labels == label).nonzero()[0])
        for label in range(max_label)
    ])

    points = np.loadtxt(points_file)

    cluster_flag = np.asarray([
        (np.linalg.norm(np.asarray(cluster.points)[:, None] - points[None, :], axis=2) < 0.005).any()
        for cluster in clusters
    ])
    cluster_flag[0] = not add_outliers  # cluster flag == true <=> outliers will be removed

    accepted_clusters = clusters[cluster_flag.nonzero()]
    rejected_clusters = clusters[(~cluster_flag).nonzero()]

    combined_rc = combine_point_clouds(rejected_clusters)
    if debug:
        combined_ac = combine_point_clouds(accepted_clusters)
        combined_ac.paint_uniform_color([0, 1, 0])
        v_combined_rc = deepcopy(combined_rc)
        v_combined_rc.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([combined_ac, v_combined_rc])
        o3d.visualization.draw_geometries([combined_rc])

    return combined_rc  # rejected clusters are the objects


ROOT_DIR = Path(__file__).parent.parent.parent.parent


def compute_masks(files, in_dir: Path, out_dir: Path, zv_points_file: Path,
                  zv_bbox_file: Path, add_outliers: bool, db_clustering: DBClustering, debug: bool):

    for file in tqdm(files):
        raw_rs_rgb, raw_rs_depth, raw_zv_rgb, raw_zv_depth, _ = DatasetInterface.load(file)
        zv_pcd = imgs_to_pcd(raw_zv_rgb, raw_zv_depth, rs_ci)

        if not zv_points_file.exists():
            # select points that belong to the background
            v = o3d.visualization.VisualizerWithEditing()
            v.create_window()
            v.add_geometry(zv_pcd)
            v.run()
            v.destroy_window()
            picked_points_indices = v.get_picked_points()
            picked_points = np.array(np.asarray(zv_pcd.points)[picked_points_indices])
            np.savetxt(zv_points_file, picked_points)

        if debug:
            o3d.visualization.draw_geometries([zv_pcd], window_name='original pcd')

        zv_pcd = crop_by_bbox(zv_pcd, zv_bbox_file, debug)
        zv_pcd = remove_clusters_by_region(zv_pcd, zv_points_file, add_outliers, debug)
        zv_pcd = db_clustering.select_largest_cluster(zv_pcd)

        if debug:
            o3d.visualization.draw_geometries([zv_pcd], window_name='final pcd')

        zv_rgb, zv_depth, zv_ul, zv_lr = pcd_to_imgs(zv_pcd, rs_ci)
        zv_rgb, zv_depth = resize(zv_rgb, zv_depth, zv_ul, zv_lr, cropped=False, resulting_shape=raw_zv_rgb.shape[:2])

        mask = np.where(np.isnan(zv_depth), False, True)

        if debug:
            rs_rgb = raw_rs_rgb * mask[..., None]
            _, ax = plt.subplots(2, 3)
            ax[0][0].imshow(to_rgb(raw_zv_rgb))
            ax[0][1].imshow(to_rgb(zv_rgb))
            ax[0][2].imshow(mask)
            ax[1][0].imshow(to_rgb(raw_rs_rgb))
            ax[1][1].imshow(to_rgb(rs_rgb))
            plt.show()

        out_path = out_dir / file.relative_to(in_dir)
        DatasetInterface.save(raw_rs_rgb, raw_rs_depth, raw_zv_rgb, raw_zv_depth, mask, out_path)


def main(args):
    in_dir = ROOT_DIR / args.in_dir
    out_dir = ROOT_DIR / args.out_dir
    zv_bbox_file = ROOT_DIR / "resources/calibration/zv_bbox.npz"
    zv_points_file = ROOT_DIR / "resources/calibration/zv_points.out"
    debug = args.debug
    jobs = args.jobs
    add_outliers = True  # add outliers to first clustering result (will be removed in 2nd clustering)

    db_clustering = DBClustering(
        max_neigbour_distance=0.15,
        min_points=50,
        eps=0.01,
        debug=debug
    )

    files = DatasetInterface.get_files_by_path(in_dir)
    random.shuffle(files)

    if jobs > 1:
        files_chunked = split(files, jobs)
        Parallel(jobs)(
            delayed(compute_masks)(files_chunk, in_dir, out_dir, zv_points_file, zv_bbox_file, add_outliers, db_clustering, debug)
            for files_chunk in files_chunked
        )

    else:
        compute_masks(files, in_dir, out_dir, zv_points_file,
                      zv_bbox_file, add_outliers, debug)


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("in_dir", type=Path, help="directory the segmentation mask should be computed for")
    argparse.add_argument("out_dir", type=Path, help="where the output files will be saved to")
    argparse.add_argument("--debug", action="store_true", help="activate to show debug visualizations")
    argparse.add_argument("--jobs", type=int, default=1, help="number of jobs (resource intensive")
    main(argparse.parse_args())
