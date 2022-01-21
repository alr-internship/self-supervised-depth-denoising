RealSense
=======
- *camera product line:* D401
- *camera product id:* 0B07
- Depth Camera Intrinsics
- *Principal Point:* 635.5166625976562 359.2284851074219
- *Focal Length:*    639.042724609375 639.042724609375
- **Color Camera Intrinsics**
- *Principal Point:* 936.4846801757812 537.48779296875
- *Focal Length:*    1377.6448974609375 1375.7239990234375
- retrieved camera relative to color camera


Zivid
========
CameraIntrinsics:
  CameraMatrix:
    CX: 951.679748535156
    CY: 594.778503417969
    FX: 2760.12109375
    FY: 2759.78198242188
  Distortion:
    K1: -0.273149877786636
    K2: 0.354378968477249
    K3: -0.322515338659286
    P1: -0.000344441272318363
    P2: 0.000198412672034465

Separated camera intrinsic parameters with description:
    CX: 951.68       x coordinate of the principal point
    CY: 594.779      y coordinate of the principal point
    FX: 2760.12      Focal length in x
    FY: 2759.78      Focal length in y
    K1: -0.27315     First radial distortion term
    K2: 0.354379     Second radial distortion term
    K3: -0.322515    Third radial distortion term
    P1: -0.000344441 First tangential distortion term
    P2: 0.000198413  Second tangential distortion term

Realsense
====
CameraMatrix
[[1.37764490e+03 0.00000000e+00 9.36484680e+02]
 [0.00000000e+00 1.37572400e+03 5.37487793e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
Distortion Coefficient
[0. 0. 0. 0. 0.]
Zivid
====
CameraMatrix
[[2.76012e+03 0.00000e+00 9.51680e+02]
 [0.00000e+00 2.75978e+03 5.94779e+02]
 [0.00000e+00 0.00000e+00 1.00000e+00]]
Distortion Coefficient
[-2.73150e-01  3.54379e-01 -3.44441e-04  1.98413e-04 -3.22515e-01]