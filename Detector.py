import cv2
import numpy as np
import sys
from vidstab import VidStab


def imread_utf(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)


def putText(img, text):
    cv2.putText(img, text, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)


def matEqualsRate(mat1, mat2):
    equalMat = mat1 == mat2
    equalMatStr = str(equalMat)
    trueCount = equalMatStr.count('True')
    falseCount = equalMatStr.count('False')
    equalRate = int(100 * (trueCount / (trueCount + falseCount)))
    return equalRate


def main(args):
    if len(args) is 1:
        args +=['3']
    normalBW = imread_utf('挂地线_Pr' + args[1] + '.png')
    normalBW = cv2.cvtColor(normalBW, cv2.COLOR_BGR2GRAY)

    videoInput = cv2.VideoCapture('挂地线_Pr.mp4')
    fps = videoInput.get(cv2.CAP_PROP_FPS)
    size = (int(videoInput.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoInput.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frameCount = videoInput.get(cv2.CAP_PROP_FRAME_COUNT)

    videoFrames = []
    while videoInput.isOpened():
        ret, frame = videoInput.read()
        if ret is False:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        videoFrames += [gray]
    videoInput.release()

    window = 'Gray equals'
    cv2.namedWindow(window)
    for i in range(60):
        startFrameIndex = int(len(videoFrames) * (i / 60))
        endFrameIndex = int(len(videoFrames) * ((i + 1) / 60))
        maxEqualsRate = 0
        for frameIndex in range(startFrameIndex, endFrameIndex):
            frame = videoFrames[frameIndex]
            equalsRate = matEqualsRate(normalBW, frame)
            if equalsRate > maxEqualsRate:
                maxEqualsRate = equalsRate
        timeStart = startFrameIndex / 25
        timeEnd = endFrameIndex / 25
        print(str(timeStart) + '-' + str(timeEnd) + '秒：' + str(maxEqualsRate), sep='')
        for frameIndex in range(startFrameIndex, endFrameIndex):
            frame = videoFrames[frameIndex]
            putText(frame, str(maxEqualsRate))
            cv2.imshow(window, frame)
            if cv2.waitKey(1) == 27:
                exit(0)


if __name__ == '__main__':
    main(sys.argv)

# window = 'Gray equals'
# cv2.namedWindow(window)
# while videoInput.isOpened():
#     ret, frame = videoInput.read()
#     # videoFrames += [frame]
#     if ret is False:
#         break
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray &= normalBW
#     # normalBW &= gray
#     equalRate= matEqualsRate(normalBW , gray)
#     if equalRate < 30:
#         drawText(gray, str(equalRate))
#     cv2.imshow(window, gray)
#     if cv2.waitKey(1) == 27:
#         break
