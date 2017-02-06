#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# this file is coming from openface: https://github.com/cmusatyalab/openface, with some changes
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import cv2
from skimage import io
import dlib
import numpy as np
import os,errno
import time

file_dir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(file_dir.replace('util', ''), 'model')
dlib_model_dir = os.path.join(modelDir, 'dlib')
dlibFacePredictor = os.path.join(dlib_model_dir, "shape_predictor_68_face_landmarks.dat")
opencv_model = './model/opencv/cascade.xml'
ts = 0.1
size = 55
ext = 'jpg'
verbose = True

"""Module for dlib-based alignment."""
TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

def mkdirP(path):
    """
    Create a directory and don't error if the path already exists.

    If the directory already exists, don't do anything.

    :param path: The directory to create.
    :type path: str
    """
    assert path is not None

    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class Image:
    """Object containing image metadata."""

    def __init__(self, name, path):
        """
        Instantiate an 'Image' object.

        :param cls: The image's class; the name of the person.
        :type cls: str
        :param name: The image's name.
        :type name: str
        :param path: Path to the image on disk.
        :type path: str
        """
        assert name is not None
        assert path is not None

        self.name = name
        self.path = path

    def getBGR(self):
        """
        Load the image from disk in BGR format.

        :return: BGR image. Shape: (height, width, 3)
        :rtype: numpy.ndarray
        """
        try:
            bgr = cv2.imread(self.path)
        except:
            bgr = None
        return bgr

    def getRGB(self):
        """
        Load the image from disk in RGB format.

        :return: RGB image. Shape: (height, width, 3)
        :rtype: numpy.ndarray
        """
        rgb = io.imread(self.path)
#        bgr = self.getBGR()
#        if bgr is not None:
#            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#        else:
#            rgb = None
        return rgb


def iterImgs(directory):
    u"""
    Iterate through the images in a directory.

    The images should be organized in subdirectories
    named by the image's class (who the person is)::

       $ tree directory
       person-1
       ── image-1.jpg
       ├── image-2.png
       ...
       └── image-p.png

       ...

       person-m
       ├── image-1.png
       ├── image-2.jpg
       ...
       └── image-q.png


    :param directory: The directory to iterate through.
    :type directory: str
    :return: An iterator over Image objects.
    """
    assert directory is not None

    exts = [".jpg", ".png"]
    fname = os.path.basename(directory)
    imageName, ext = os.path.splitext(fname)
    if ext in exts:
        return Image('temp1', directory)


class AlignDlib:
    """
    Use `dlib's landmark estimation <http://blog.dlib.net/2014/08/real-time-face-pose-estimation.html>`_ to align faces.

    The alignment preprocess faces for input into a neural network.
    Faces are resized to the same size (such as 96x96) and transformed
    to make landmarks (such as the eyes and nose) appear at the same
    location on every image.

    Normalized landmarks:

    .. image:: ../images/dlib-landmark-mean.png
    """

    #: Landmark indices corresponding to the inner eyes and bottom lip.
    INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]

    def __init__(self, facePredictor):
        """
        Instantiate an 'AlignDlib' object.

        :param facePredictor: The path to dlib's
        :type facePredictor: str
        """
        assert facePredictor is not None

#        self.detector = dlib.get_frontal_face_detector()
        self.predictor = facePredictor

    def getAllFaceBoundingBoxes(self, rgbImg):
        """
        Find all face bounding boxes in an image.

        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :return: All face bounding boxes in an image.
        :rtype: dlib.rectangles
        """
        assert rgbImg is not None

        try:
            return self.detector(rgbImg, 1)
        except Exception as e:
            print("Warning: {}".format(e))
            # In rare cases, exceptions are thrown.
            return []

    def getLargestFaceBoundingBox(self, rgbImg):
        """
        Find the largest face bounding box in an image.

        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :return: The largest face bounding box in an image, or None.
        :rtype: dlib.rectangle
        """
        assert rgbImg is not None

        faces = self.getAllFaceBoundingBoxes(rgbImg)
        if len(faces) > 0:
            return max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            return None

    def findLandmarks(self, rgbImg, bb):
        """
        Find the landmarks of a face.

        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :param bb: Bounding box around the face to find landmarks for.
        :type bb: dlib.rectangle
        :return: Detected landmark locations.
        :rtype: list of (x,y) tuples
        """
        assert rgbImg is not None
        assert bb is not None

        points = self.predictor(rgbImg, bb)
        return list(map(lambda p: (p.x, p.y), points.parts()))

    def align(self, imgDim, rgbImg, bb=None, ts=None,landmarkIndices=INNER_EYES_AND_BOTTOM_LIP, opencv_model=None):
        r"""align(imgDim, rgbImg, bb=None, landmarks=None, landmarkIndices=INNER_EYES_AND_BOTTOM_LIP)

        Transform and align a face in an image.

        :param imgDim: The edge length in pixels of the square the image is resized to.
        :type imgDim: int
        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :param bb: Bounding box around the face to align. \
                   Defaults to the largest face.
        :type bb: dlib.rectangle
        :param pad: padding bb by left, top, right, bottom
        :type pad: list
        :param landmarks: Detected landmark locations. \
                          Landmarks found on `bb` if not provided.
        :type landmarks: list of (x,y) tuples
        :param landmarkIndices: The indices to transform to.
        :type landmarkIndices: list of ints
        :return: The aligned RGB image. Shape: (imgDim, imgDim, 3)
        :rtype: numpy.ndarray
        """
        assert imgDim is not None
        assert rgbImg is not None
        assert landmarkIndices is not None
#        bb = self.getLargestFaceBoundingBox(rgbImg)
        face_cascade = cv2.CascadeClassifier(opencv_model)
        faces = face_cascade.detectMultiScale(rgbImg, 1.1, 2, minSize=(30, 30))
        dlib_rects = []
        for (x,y,w,h) in faces:
            dlib_rects.append(dlib.rectangle(int(x), int(y), int(x+w), int(y+h)))
            if len(faces) > 0:
                bb = max(dlib_rects, key=lambda rect: rect.width() * rect.height())
            else:
                bb = None
        if bb is None:
            return

        landmarks = self.findLandmarks(rgbImg, bb)

        npLandmarks = np.float32(landmarks)
        npLandmarkIndices = np.array(landmarkIndices)

        dstLandmarks = imgDim * MINMAX_TEMPLATE[npLandmarkIndices]
        if ts is not None:
            # reserve more area of forehead on a face
            dstLandmarks[(0,1),1] = dstLandmarks[(0,1),1] + imgDim * float(ts)
            dstLandmarks[2,1] = dstLandmarks[2,1] + imgDim * float(ts) / 2
        H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices],dstLandmarks)
        return cv2.warpAffine(rgbImg, H, (imgDim, imgDim))


def alignMain(inputraw, outputdir, predict):
    mkdirP(outputdir)
    imgObject = iterImgs(inputraw)
#    print imgs
    # Shuffle so multiple versions can be run at once.
    landmarkIndices = AlignDlib.INNER_EYES_AND_BOTTOM_LIP
    align = AlignDlib(predict)

#        print("=== {} ===".format(imgObject.path))
    outputPrefix = os.path.join(outputdir, imgObject.name)
    imgName = outputPrefix + "." + ext

    rgb = imgObject.getRGB()
    if rgb is None:
        print("Unable to load.")
        outRgb = None
    else:
        outRgb = align.align(size, rgb, ts=ts, landmarkIndices=landmarkIndices,
                             opencv_model=opencv_model)
        if outRgb is None and verbose:
            return "Unable_align"

    if outRgb is not None:
#        outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
        io.imsave(imgName, outRgb)
        return imgName
