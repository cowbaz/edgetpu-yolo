import datetime
import numpy as np
from picamera2 import Picamera2


class picam2:
    def __init__(
        self,
        video_main={
            "size": (1280, 720),
            "format": "RGB888",
        },  # widht, height & format of video main
        video_lores={
            "size": (320, 240),
            "format": "YUV420",
        },  # widht, height & format of video lores
        capture_main={"size": (1280, 720)},
    ):  # widht, height & format of capture main
        self.camera = Picamera2()

        self.video_config = self.camera.create_video_configuration(
            main=video_main, lores=video_lores
        )

        self.capture_config = self.camera.create_still_configuration(main=capture_main)

        self.init_config = self.camera.configure(
            self.video_config
        )  # set the initial config as video
        self.start = self.camera.start()

    def motion_detection(self):
        """
        Capture frames from camera and detect motion
        Args: None

        Returns:
        True for motion detected
        False for motion not detected
        """

        lsize = (320, 240)
        w, h = lsize
        prev = None
        motion_detected = False

        while True:
            cur = self.camera.capture_buffer("lores")
            cur = cur[: w * h].reshape(h, w)
            if prev is not None:
                # Measure pixels differences between current and
                # previous frame
                mse = np.square(np.subtract(cur, prev)).mean()
                if mse > 7:
                    # picam2.stop()
                    motion_detected = True
                    current_time = datetime.datetime.now()
                    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"New Motion Detected {formatted_time}, mse: ", mse)
                    return motion_detected
                else:
                    # picam2.stop()
                    current_time = datetime.datetime.now()
                    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Motion Not Detected {formatted_time}, mse: ", mse)
                    return motion_detected
            prev = cur

    def capture_image(self):
        """
        Capture image in numpy array format
        Args:
        None

        Returns:
        image numpy array in shape of (hight, width, channel)
        """
        # Switch to capture mode
        self.camera.switch_mode(self.capture_config)
        image_array = self.camera.capture_array("main")

        # Switch back to detection video mode
        self.camera.switch_mode(self.video_config)

        return image_array

    def close(self):
        self.camera.close()
