import cv2
import numpy as np
import Homography as homo


def imread(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)


stableBW = imread('挂地线_Pr1.png')
stableBW = cv2.cvtColor(stableBW, cv2.IMREAD_COLOR)
videoInput = cv2.VideoCapture('挂地线_Pr.mp4')
fps = videoInput.get(cv2.CAP_PROP_FPS)
size = (int(videoInput.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoInput.get(cv2.CAP_PROP_FRAME_HEIGHT)))
frameCount = videoInput.get(cv2.CAP_PROP_FRAME_COUNT)
videoOutput = cv2.VideoWriter('stable.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size, False)  # MPEG-4编码
while videoInput.isOpened():
    ret, frame = videoInput.read()
    if ret is False:
        break
    frame = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    videoOutput.write(homo.alignImages(stableBW, frame)[0])
videoInput.release()
videoOutput.release()
