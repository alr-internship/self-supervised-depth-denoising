import zivid
import datetime
import cv2

class Zivid:

    def __init__(self) -> None:
        app = zivid.Application()

        print("Connecting to camera")
        self.camera = app.connect_camera()

        print("Configuring settings")
        self.settings = zivid.Settings()
        self.settings.acquisitions.append(zivid.Settings.Acquisition())
        self.settings.acquisitions[0].aperture = 5.66
        self.settings.acquisitions[0].exposure_time = datetime.timedelta(microseconds=6500)
        self.settings.processing.filters.outlier.removal.enabled = True
        self.settings.processing.filters.outlier.removal.threshold = 5.0        

    def collect_frame(self):
        print("Capturing frame")
        with self.camera.capture(self.settings) as frame:
            data_file = "Frame.zdf"
            print(f"Saving frame to file: {data_file}")
            frame.save(data_file)

            rgb_image = Zivid._convert_2_2d(point_cloud=frame)

            return rgb_image


    def _convert_2_2d(point_cloud):
        """Convert from point cloud to 2D image.
        Args:
            point_cloud: a handle to point cloud in the GPU memory
        Returns rgb opencv image 
        """
        rgba = point_cloud.copy_data("rgba")
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        return bgr