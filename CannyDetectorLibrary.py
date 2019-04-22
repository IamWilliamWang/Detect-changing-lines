import cv2
import numpy


class VideoFileUtil:

    @staticmethod
    def OpenVideos(inputVideoFilename=None, outputVideoFilename=None, outputVideoEncoding='DIVX'):  # MPEG-4编码
        '''
        打开输入输出视频文件
        :param inputVideoFilename: 输入文件名
        :param outputVideoFilename: 输出文件名
        :param outputVideoEncoding: 输出文件的视频编码
        :return: 输入输出文件流
        '''
        videoInput = None
        videoOutput = None
        if inputVideoFilename is not None:
            videoInput = VideoFileUtil.OpenInputVideo(inputVideoFilename)  # 打开输入视频文件
        if outputVideoFilename is not None:
            videoOutput = VideoFileUtil.OpenOutputVideo(outputVideoFilename, videoInput, outputVideoEncoding)
        return videoInput, videoOutput

    @staticmethod
    def OpenInputVideo(inputVideoFilename):
        '''
        打开输入视频文件
        :param inputVideoFilename: 输入文件名
        :return: 输入文件流
        '''
        return cv2.VideoCapture(inputVideoFilename)

    @staticmethod
    def OpenOutputVideo(outputVideoFilename, inputFileStream, outputVideoEncoding='DIVX'):
        '''
        打开输出视频文件
        :param outputVideoFilename: 输出文件名
        :param inputFileStream: 输入文件流（用户获得视频基本信息）
        :param outputVideoEncoding: 输出文件编码
        :return: 输出文件流
        '''
        # 获得码率及尺寸
        fps = int(inputFileStream.get(cv2.CAP_PROP_FPS))
        size = (int(inputFileStream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(inputFileStream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return cv2.VideoWriter(outputVideoFilename, cv2.VideoWriter_fourcc(*outputVideoEncoding), fps, size,
                               False)

    @staticmethod
    def CloseVideos(inputVideoStream=None, outputVideoStream=None):
        '''
        关闭输入输出视频文件
        :param inputVideoStream: 输入文件流
        :param outputVideoStream: 输出文件流
        :return:
        '''
        if inputVideoStream is not None:
            inputVideoStream.release()
        if outputVideoStream is not None:
            outputVideoStream.release()


class Transformer:
    '''
    图像转换器，负责图像的读取，变灰度，边缘检测和线段识别。
    《请遵守以下命名规范：前缀image、img代表彩色图。前缀为gray代表灰度图。前缀为edges代表含有edge的黑白图。前缀为lines代表edges中各个线段的结构体。前缀为static代表之后的比较要以该变量为基准进行比较。可以有双前缀》
    '''

    @staticmethod
    def Imread(filename_unicode):
        '''
        读取含有unicode文件名的图片
        :param filename_unicode: 含有unicode的图片名
        :return:
        '''
        return cv2.imdecode(numpy.fromfile(filename_unicode, dtype=numpy.uint8), -1)

    @staticmethod
    def IsGrayImage(grayOrImg):
        '''
        检测是否为灰度图，灰度图为True，彩图为False
        :param grayOrImg: 图片
        :return: 是否为灰度图
        '''
        return len(grayOrImg.shape) is 2

    @staticmethod
    def GetGrayFromBGRImage(image):
        '''
        将读取的BGR转换为单通道灰度图
        :param image: BGR图片
        :return: 灰度图
        '''
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def GetEdgesFromGray(grayFrame):
        '''
        将灰度图调用canny检测出edges，返回灰度edges图
        :param grayFrame: 灰度图
        :return: 含有各个edges的黑白线条图
        '''
        grayFrame = cv2.GaussianBlur(grayFrame, (3, 3), 0)  # 高斯模糊，去除图像中不必要的细节
        edges = cv2.Canny(grayFrame, 50, 150, apertureSize=3)
        return edges

    @staticmethod
    def GetEdgesFromImage(imageBGR):
        '''
        将彩色图转变为带有所有edges信息的黑白线条图
        :param imageBGR: 彩色图
        :return:
        '''
        return Transformer.GetEdgesFromGray(Transformer.GetGrayFromBGRImage(imageBGR))

    @staticmethod
    def GetLinesFromEdges(edgesFrame, threshold=200):
        '''
        单通道灰度图中识别内部所有线段并返回
        :param edgesFrame: edges图
        :param threshold: 阈值限定，线段越明显阈值越大。小于该阈值的线段将被剔除
        :return:
        '''
        return cv2.HoughLines(edgesFrame, 1, numpy.pi / 180, threshold)


class PlotUtil:
    '''
    用于显示图片的帮助类。可以在彩图中画霍夫线
    '''

    @staticmethod
    def PaintLinesOnImage(img, houghLines, paintLineCount=1):
        '''
        在彩色图中划指定条霍夫线，线段的优先级由长到短
        :param img: BGR图片
        :param houghLines: 霍夫线，即HoughLines函数返回的变量
        :param paintLineCount: 要画线的个数
        :return:
        '''
        for i in range(paintLineCount):
            for rho, theta in houghLines[i]:
                a = numpy.cos(theta)
                b = numpy.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * a)
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    @staticmethod
    def PutText(img, text, location=(30, 30)):
        '''
        在彩图img上使用默认字体写字
        :param img: 需要放置文字的图片
        :param text: 要写上去的字
        :param location: 字的位置
        :return:
        '''
        cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)


class Detector:
    def __init__(self):
        self.__firstFramePosition = None  # 处理视频文件时记录的当前片段在视频中的开始帧号
        self.__lastFramePosition = None  # 处理视频文件时记录的当前片段在视频中的结束帧号
        self.__originalFrames = None  # 处理视频流时记录当前片段的原始录像
        self.__showWarningMutex = 0  # 用于切换警报状态的信号量

    def LinesEquals(self, lines1, lines2, comparedLinesCount):
        '''
        HoughLines函数返回的lines判断是否相等
        :param lines1: 第一个lines
        :param lines2: 第二个lines
        :param comparedLinesCount: 比较前几条line
        :return: 是否二者相等
        '''
        if lines1 is None or lines2 is None:
            return False
        sameCount = 0
        diffCount = 0
        try:
            for i in range(comparedLinesCount):
                for rho1, theta1 in lines1[i]:
                    for rho2, theta2 in lines2[i]:
                        if rho1 != rho2 or theta1 != theta2:
                            diffCount += 1
                        else:
                            sameCount += 1
        except IndexError:  # 阈值过高的话会导致找不到那么多条line，报错可以忽略
            pass
        return sameCount / (sameCount + diffCount) > 0.9  # 不同到一定程度再报警

    def GetNoChangeEdges_fromVideo(self, videoFilename, startFrameRate=0., endFrameRate=1., outputEdgesFilename=None):
        '''
        从视频文件中提取不动物体的帧
        :param videoFilename: 文件名
        :param startFrameRate: 开始读取帧处于视频的比例，必须取0-1之间
        :param endFrameRate: 结束读取帧处于视频的比例，必须取0-1之间
        :param outputEdgesFilename: EdgesFrame全部输出到视频为该名的文件中（测试时用）
        :return: 不动物体的Edges帧
        '''
        # 打开输入输出视频文件
        videoInput = VideoFileUtil.OpenInputVideo(videoFilename)
        frame_count = videoInput.get(cv2.CAP_PROP_FRAME_COUNT)  # 获取视频总共的帧数
        outputVideo = None  # 声明输出文件
        if outputEdgesFilename is not None:
            outputVideo = VideoFileUtil.OpenOutputVideo(outputEdgesFilename, videoInput)
        staticEdges = None  # 储存固定的Edges
        videoInput.set(cv2.CAP_PROP_POS_FRAMES, int(frame_count * startFrameRate))  # 指定读取的开始位置
        self.__firstFramePosition = int(frame_count * startFrameRate)  # 记录第一帧的位置
        self.__lastFramePosition = int(frame_count * endFrameRate)  # 记录最后一帧的位置
        if endFrameRate != 1:  # 如果提前结束，则对总帧数进行修改
            frame_count = int(frame_count * (endFrameRate - startFrameRate))
        while videoInput.isOpened() and frame_count >= 0:  # 循环读取
            ret, frame = videoInput.read()
            if ret is False:
                break
            edges = Transformer.GetEdgesFromImage(frame)  # 对彩色帧进行边缘识别
            if staticEdges is None:
                staticEdges = edges  # 初始化staticEdges
            else:
                staticEdges &= edges  # 做与运算，不同点会被去掉
            if outputEdgesFilename is not None:
                outputVideo.write(edges)  # 写入边缘识别结果
            frame_count -= 1
            VideoFileUtil.CloseVideos(videoInput, outputVideo)
        return staticEdges

    def GetNoChangeEdges_fromSteam(self, inputStream, frame_count=20, outputEdgesFilename=None):
        '''
        从输入流中提取不动物体的Edges帧
        :param inputStream: 输入文件流
        :param frame_count: 要读取的帧数
        :param outputEdgesFilename: EdgesFrame全部输出到视频为该名的文件中（测试用）
        :return: 不动物体的Edges帧、原本的彩色帧组
        '''
        outputVideo = None
        if outputEdgesFilename is not None:
            outputVideo = VideoFileUtil.OpenOutputVideo(outputEdgesFilename, inputStream)
        staticEdges = None
        self.__originalFrames = []
        while inputStream.isOpened() and frame_count >= 0:
            ret, frame = inputStream.read()
            if ret is False:
                break
            self.__originalFrames += [frame]
            edges = Transformer.GetEdgesFromImage(frame)  # 边缘识别
            if staticEdges is None:
                staticEdges = edges  # 初始化staticEdges
            else:
                staticEdges &= edges  # 做与运算，不同点会被去掉
            if outputEdgesFilename is not None:
                outputVideo.write(edges)  # 写入边缘识别结果
            frame_count -= 1
        return staticEdges

    def ReadFrames(self, stream, readFramesCount):
        '''
        从输入流中读取readFramesCount个帧并返回，如果没有读取则返回None
        :param stream: 输入流
        :param readFramesCount: 要读取的帧数
        :return:
        '''
        frames = []
        while stream.isOpened():
            ret, frame = stream.read()
            if ret is False:
                break
            frames += [frame]
            if len(frames) >= readFramesCount:
                break
        if len(frames) is 0:
            return None
        return frames

    def IsWarningStatusChanged(self, exceptionOccurred, consecutiveOccurrencesNumber=3):
        '''
        显示warning状态是否需要改变，True为需要显示Warning。False为需要关闭Warning。None为保持不变
        :param exceptionOccurred: 是否发生异常
        :param consecutiveOccurrencesNumber: 连续几次同样时间发生后给予改变当前警报状态的指示
        :return:
        '''
        if exceptionOccurred:  # 如果发生异常
            if self.__showWarningMutex < 0:  # 清除信号量向正方向
                self.__showWarningMutex = 0
            else:  # 在正方向，则增添信号量
                self.__showWarningMutex += 1
            if self.__showWarningMutex > (consecutiveOccurrencesNumber - 1):
                return True  # 连续3次就返回显示warning
        else:
            if self.__showWarningMutex > 0:
                self.__showWarningMutex = 0
            else:
                self.__showWarningMutex -= 1
            if self.__showWarningMutex < -(consecutiveOccurrencesNumber - 1):
                return False  # 连续3次就返回撤销warning
        return None
