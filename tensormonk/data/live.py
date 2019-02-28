""" TensorMONK :: data :: live """

import time
import numpy as np
from threading import Thread
import cv2


class Camera:
    r""" Extended on imutils.WebcamVideoStream with frame averaging (can be
    used as an iterator) """
    live = False
    frame = None
    video = None

    def __init__(self,
                 gizmo: int = 0,
                 rgb: bool = False,
                 mirror: bool = False,
                 average_frames: int = 1):

        # additional args
        self.rgb = rgb
        self.mirror = mirror
        self.average_frames = min(6, average_frames)

        # camera
        Camera.video = cv2.VideoCapture(gizmo)
        Camera.live = True
        self._grab_a_frame()
        while Camera.frame is None:
            time.sleep(0.)
            self._grab_a_frame()

    def initialize(self):
        Thread(target=self._continuous_feed, args=()).start()
        return self

    def _continuous_feed(self):
        while True:
            if not Camera.live:
                return
            self._grab_a_frame()

    def _grab_a_frame(self):
        if self.average_frames == 1:
            frame = Camera.video.read()[1]
        else:
            # frame averaging to minimize motion blur
            accumulate = None
            for _ in range(self.average_frames):
                frame = Camera.video.read()[1].astype(np.float32)[1]
                if accumulate is None:
                    accumulate = frame
                else:
                    accumulate += frame
            frame = (accumulate / self.average_frames).astype(np.uint8)
        if self.rgb:
            frame = frame[:, :, [2, 1, 0]]
        if self.mirror:
            frame = frame[:, ::-1, :]
        Camera.frame = frame

    def __iter__(self):
        return self

    def __next__(self):
        return self.read()

    def read(self):
        return Camera.frame

# from matplotlib import pyplot as plt
# camera = Camera().initialize()
# %timeit camera.read()
# %timeit next(camera)
# plt.imshow(camera.read())
