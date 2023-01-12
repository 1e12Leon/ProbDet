import time

import cv2
import numpy as np
from PIL import Image

from yolov7.yolo import YOLO
from CenterNet.centernet import CenterNet

if __name__ == '__main__':
    centernet = CenterNet()
    yolo = YOLO()
    crop = False
    count = False

    # img = input('Input image filename:')
    img = 'D:\Deep_Learning_folds\ProbEn\yolov7\img\street.jpg'
    try:
        image1 = Image.open(img)
        image2 = Image.open(img)
    except:
        print('Open Error! Try again!')
    else:
        print("-----------------")
        print("yolov7:")
        r_image = yolo.detect_image(image1, crop=crop, count=count)
        r_image.show()
        # dets_yolo, scores_yolo = yolo.detect_image_dets(image1)
        # print(dets_yolo)
        # print(scores_yolo)

        print("-----------------")
        print("centernet:")
        r_image2 = centernet.detect_image(image2, crop = crop, count=count)
        r_image2.show()
        # dets_centernet, scores_centernet = centernet.detect_image_dets(image2)
        # print(dets_centernet)
        #print(scores_centernet)