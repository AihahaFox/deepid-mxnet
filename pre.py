# -*- coding: utf-8 -*-
import os
import run
import dlib
import zmq
import cv2
import numpy as np
import cv
import base64
import tools


def deepid_init():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    modelDir = os.path.join(file_dir.replace('util', ''), 'model')
    dlib_model_dir = os.path.join(modelDir, 'dlib')
    dlibFacePredictor = os.path.join(dlib_model_dir, "shape_predictor_68_face_landmarks.dat")
    predict = dlib.shape_predictor(dlibFacePredictor)
    return predict


def CleanDir(Dir):
    if os.path.isdir(Dir):
        paths = os.listdir(Dir)
        for path in paths:
            filePath = os.path.join(Dir, path)
            if os.path.isfile(filePath):
                os.remove(filePath)


def decode_base64(data):
    """Decode base64, padding being optional.
    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    missing_padding = 4 - len(data) % 4
    if missing_padding:
        data += (b'=' * missing_padding)
    print len(data)
    return base64.b64decode(data)

if __name__ == "__main__":
    predict = deepid_init()
    #  Prepare our context and sockets
    context = zmq.Context()
    socket_rev = context.socket(zmq.PULL)
    try:
        socket_rev.connect("tcp://192.168.1.212:4130")
    except:
        print zmq.ZMQError
    print 'rev success'

    socket_res = context.socket(zmq.PUSH)
    try:
        b = socket_res.connect("tcp://192.168.1.212:4131")
    except:
        print zmq.ZMQError

    print 'res success'
    result = []
    send_message = []
    fe_list = []
    while True:
        # 用了NOBLOCK，就意味着得不到消息时不会堵塞在这里
        while True:
            lenth = len(os.listdir('people_image'))
            try:
                recv_message = socket_rev.recv(zmq.NOBLOCK)
                a = recv_message.split(' ')
                bone_ID = int(a[0])
                isValid = int(a[1])
                imageSize = int(a[2])
                height = int(a[3])
                width = int(a[4])
                depth = int(a[5])
                nChannels = int(a[6])
                widthStep = int(a[7])
                strImg = a[8]
                print 'bone_ID = {} isValid = {} imageSize = {} height = {} width = {} depth = {} nChannels = {} widthStep = {}'.format(bone_ID, isValid, imageSize, height, width, depth, nChannels, widthStep)
                strImg = strImg[:imageSize/3*4]
                decode_str = decode_base64(strImg)
                pImg = cv.CreateImageHeader((width, height), depth, nChannels)
                cv.SetData(pImg, decode_str, widthStep)
                img = np.asarray(cv.GetMat(pImg))
                result.append(img)
                if socket_rev.getsockopt(zmq.EVENTS) == 0:
                    break
            except zmq.ZMQError:
                pass
        for i in range(len(result)):
            cv2.imwrite('temp{}.jpg'.format(i), result[i])
            f, fe = run.deepid('temp{}.jpg'.format(i), predict)
            fe_list.append(fe)
            send_message.append(f)
        result = []
        print send_message
        dic = {i: send_message.count(i) for i in send_message}
        fname = max(dic, key=dic.get)
        # if (fname == 'x_{}'.format(lenth-1)) and 'unknow_person' in dic.keys():
        #     socket_res.send(str(bone_ID) + ' ' + 'unknow_person')
        #     print 'unknow_person'
        # else:
        #     print fname
        #     socket_res.send(str(bone_ID) + ' ' + fname)
        if fname == 'unknow_person':
            for i in range(len(send_message)):
                if send_message[i] == 'Unable_align':
                    continue
                tools.tools.save_image('temp{}.jpg'.format(i), './people_image/x_'+str(lenth), fe_list[i])
        elif fname == 'Unable_align':
            pass
        else:
            for i in range(len(send_message)):
                if send_message[i] == 'Unable_align':
                    continue
                tools.tools.save_image('temp{}.jpg'.format(i), './people_image/'+fname, fe_list[i])
        print fname
        socket_res.send(str(bone_ID) + ' ' + fname)
        send_message = []
        fe_list = []
    context.term()
