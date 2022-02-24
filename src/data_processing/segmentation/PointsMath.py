"""
Main task: Point-related math
@author: GeoLab
"""
import numpy as np



def angle2p(N1, N2):
    # Input two normals, return the angle
    dt = N1[0] * N2[0] + N1[1] * N2[1] + N1[2] * N2[2]
    dt = np.arccos(np.clip(dt, -1, 1))
    r_Angle = np.degrees(dt)
    return r_Angle