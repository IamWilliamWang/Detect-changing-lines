import cv2
import numpy as np
import Homography


class Detector:
    @staticmethod
    def Imread(filename):
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)

    @staticmethod
    def PutText(img, text):
        cv2.putText(img, text, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)

    @staticmethod
    def MatrixEqualsRate(mat1, mat2):
        equalMat = mat1 == mat2
        equalMatStr = str(equalMat)
        trueCount = equalMatStr.count('True')
        falseCount = equalMatStr.count('False')
        equalRate = int(100 * (trueCount / (trueCount + falseCount)))
        return equalRate

    def Start_FileStream(self):
        normalBW = Detector.Imread('挂地线_Pr1.png')
        normalBW = cv2.cvtColor(normalBW, cv2.IMREAD_COLOR)

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
            videoFrames += [frame]
        videoInput.release()

        window = 'equals'
        cv2.namedWindow(window)
        for i in range(60):
            startFrameIndex = int(len(videoFrames) * (i / 60))
            endFrameIndex = int(len(videoFrames) * ((i + 1) / 60))
            maxEqualsRate = 0
            for frameIndex in range(startFrameIndex, endFrameIndex):
                frame, _ = Homography.alignImages(normalBW, videoFrames[frameIndex])
                # cv2.imshow('origin', frame)
                # cv2.waitKey(1)
                equalsRate = Detector.MatrixEqualsRate(normalBW, frame)
                if equalsRate > maxEqualsRate:
                    maxEqualsRate = equalsRate
            timeStart = startFrameIndex / 25
            timeEnd = endFrameIndex / 25
            print(str(timeStart) + '-' + str(timeEnd) + '秒：' + str(maxEqualsRate), sep='')
            for frameIndex in range(startFrameIndex, endFrameIndex):
                frame = videoFrames[frameIndex]
                Detector.PutText(frame, str(maxEqualsRate))
                cv2.imshow(window, frame)
                if cv2.waitKey(1) == 27:
                    exit(0)

    def Start_VideoStream(self):
        pass


if __name__ == '__main__':
    Detector().Start_FileStream()

'''
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
    equalRate= matEqualsRate(normalBW , gray)
    if equalRate < 30:
        drawText(gray, str(equalRate))
    cv2.imshow(window, gray)
    if cv2.waitKey(1) == 27:
        break
'''
