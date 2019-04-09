import cv2
import numpy as np


def imread(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)


def putText(img, text):
    cv2.putText(img, text, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)


normalBW = imread('没挂地线.png')
normalBW = cv2.cvtColor(normalBW, cv2.COLOR_BGR2GRAY)
videoInput = cv2.VideoCapture('挂地线.mp4')
# videoFrames = []
window = 'Gray equals'
cv2.namedWindow(window)
while videoInput.isOpened():
    ret, frame = videoInput.read()
    # videoFrames += [frame]
    if ret is False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray &= normalBW
    # normalBW &= gray
    equalMat = normalBW == gray
    equalMatStr = str(equalMat)
    trueCount = equalMatStr.count('True')
    falseCount = equalMatStr.count('False')
    equalRate = int(100 * (trueCount / (trueCount + falseCount)))
    putText(gray, str(equalRate))
    cv2.imshow(window, gray)
    if cv2.waitKey(1) == 27:
        break
