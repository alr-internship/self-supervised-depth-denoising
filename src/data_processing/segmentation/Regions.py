"""
Main task: Based on open3D, given 3D point clouds and several thresholds,
           generate several clusters. (This is a basic version of region growing )
           Please cite the following paper (related to this topic), if this code helps you in your research.
           @article{xia2020geometric,
                    title={Geometric primitives in LiDAR point clouds: A review},
                    author={Xia, Shaobo and Chen, Dong and Wang, Ruisheng and Li, Jonathan and Zhang, Xinchang},
                    journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
                    volume={13},
                    pages={685--707},
                    year={2020},
                    publisher={IEEE}
                   }
@author: GeoMatrix Lab
# https://github.com/GeoVectorMatrix/Open3D_Based_Py
"""
import open3d as o3d
from PointsMath import *
import numpy as np
import math
import os
import sys

class RegionGrowing:
    def __init__(self):
        """
           Init parameters
        """
        self.pcd = None  # input point clouds
        self.NPt = 0  # input point clouds
        self.nKnn = 20  # normal estimation using k-neighbour
        self.nRnn = 0.1  # normal estimation using r-neighbour
        self.rKnn = 20  # region growing using k-neighbour
        self.rRnn = 0.1  # region growing using r-neighbour
        self.pcd_tree = None  # build kdtree
        self.TAngle = 5.0
        self.Clusters = []
        self.minCluster = 100  # minimal cluster size

    def SetDataThresholds(self, pc, t_a=10.0):
        self.pcd = pc
        self.TAngle = t_a
        self.NPt = len(self.pcd.points)
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)

    def RGKnn(self):
        """
        Region growing with KNN-neighbourhood while searching
        return: a list of clusters after region growing
        """
        # Are the normals should be re-estimated?
        if len(self.pcd.normals) < self.NPt:
            self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.nRnn, max_nn=self.nKnn))
        processed = np.full(self.NPt, False)
        for i in range(self.NPt):
            if processed[i]:
                continue
            seed_queue = []
            sq_idx = 0
            seed_queue.append(i)
            processed[i] = True
            while sq_idx < len(seed_queue):
                queryPt = self.pcd.points[seed_queue[sq_idx]]
                thisNormal = self.pcd.normals[seed_queue[sq_idx]]
                [k, idx, _] = self.pcd_tree.search_knn_vector_3d(queryPt, self.rKnn)
                idx = idx[1:k]  # indexed point itself
                theseNormals = np.asarray(self.pcd.normals)[idx, :]
                for j in range(len(theseNormals)):
                    if processed[idx[j]]:  # Has this point been processed before ?
                        continue
                    thisA = angle2p(thisNormal, theseNormals[j])
                    if thisA < self.TAngle:
                        seed_queue.append(idx[j])
                        processed[idx[j]] = True
                sq_idx = sq_idx + 1
            if len(seed_queue) > self.minCluster:
                self.Clusters.append(seed_queue)

    def ReLabeles(self):
        # Based on the generated clusters, assign labels to all points
        labels = np.zeros(self.NPt) # zero = other-clusters
        for i in range(len(self.Clusters)):
            for j in range(len(self.Clusters[i])):
                labels[self.Clusters[i][j]] = i+1
        return  labels


