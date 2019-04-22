import cv2
import numpy as np
import Homography
import CannyDetectorLibrary as lib


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
        normalEdge = lib.Transformer.GetEdgesFromImage(normalBW)
        videoInput = cv2.VideoCapture('挂地线_Pr.mp4')
        fps = videoInput.get(cv2.CAP_PROP_FPS)
        size = (int(videoInput.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(videoInput.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frameCount = videoInput.get(cv2.CAP_PROP_FRAME_COUNT)
        # 读取所有帧
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
                frame, _ = Homography.alignImages(normalEdge,
                                                  lib.Transformer.GetEdgesFromImage(videoFrames[frameIndex]))
                # cv2.imshow('edges', frame)
                # cv2.waitKey(1)
                equalsRate = Detector.MatrixEqualsRate(normalEdge, frame)
                if equalsRate > maxEqualsRate:
                    maxEqualsRate = equalsRate
                break
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
        # self.CompareErrorsAndNormals()
        self.CompareErrorsAndNormals()

    def GoodMatch(self, original, image_to_compare):
        original = cv2.imread(original)
        image_to_compare = cv2.imread(image_to_compare)

        # 裁剪到挂地线部分
        original = lib.Transformer.GetEdgesFromImage(original[500:1400, 800:1600])
        image_to_compare = lib.Transformer.GetEdgesFromImage(image_to_compare[500:1400, 800:1600])

        # 1) Check if 2 images are equals
        # if original.shape == image_to_compare.shape:
        #     print("The images have same size and channels")
        #     difference = cv2.subtract(original, image_to_compare)
        #     b, g, r = cv2.split(difference)
        #
        #     if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        #         print("The images are completely Equal")
        #     else:
        #         print("The images are NOT equal")

        # 2) Check for similarities between the 2 images
        sift = cv2.xfeatures2d.SIFT_create()
        kp_1, desc_1 = sift.detectAndCompute(original, None)
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc_1, desc_2, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.99 * n.distance:
                good_points.append(m)

        # Define how similar they are
        number_keypoints = 0
        if len(kp_1) <= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)
        return len(good_points) / number_keypoints * 100
        # print("Keypoints 1ST Image: " + str(len(kp_1)))
        # print("Keypoints 2ND Image: " + str(len(kp_2)))
        # print("GOOD Matches:", len(good_points))
        # print("How good it's the match: ", len(good_points) / number_keypoints * 100)

        # result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)

        # cv2.imshow("result", cv2.resize(result, None, fx=0.4, fy=0.4))
        # cv2.imwrite("feature_matching.jpg", result)
        #
        # cv2.imshow("Original", cv2.resize(original, None, fx=0.4, fy=0.4))
        # cv2.imshow("Duplicate", cv2.resize(image_to_compare, None, fx=0.4, fy=0.4))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def CompareErrorsAndNormals(self):
        errorList = ['error.png']
        normalList = ['normal.png']
        errorList += ['error' + str(i) + '.png' for i in range(2, 7)]
        normalList += ['normal' + str(i) + '.png' for i in range(2, 8)]
        correctCount = 0
        incorrectCount = 0
        for error in errorList:
            for normal in normalList:
                print(self.GoodMatch(normal, error))
        print('------------------------------------------')
        for normal1 in normalList:
            for normal2 in normalList:
                print(self.GoodMatch(normal1, normal2))


if __name__ == '__main__':
    Detector().Start_VideoStream()

'''
@Deprecated code
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
