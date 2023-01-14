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
    img = 'img/1.jpg'
    try:
        image1 = Image.open(img)
        image2 = Image.open(img)
    except:
        print('Open Error! Try again!')
    else:
        print("-----------------")
        print("yolov7:")
        #r_image = yolo.detect_image(image1, crop=crop, count=count)
        #r_image.show()
        # tact_time = yolo.get_FPS(image1, test_interval=100)
        # print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')
        dets_yolo, scores_yolo = yolo.detect_image_dets(image1)
        t1 = time.time()
        for _ in range(100):
            dets_yolo, scores_yolo = yolo.detect_image_dets(image1)
        t2 = time.time()
        tact_time = (t2 - t1) / 100
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')
        # print(dets_yolo)
        # print(scores_yolo)

        print("-----------------")
        print("centernet:")
        # r_image2 = centernet.detect_image(image2, crop = crop, count=count)
        # r_image2.show()
        # tact_time = centernet.get_FPS(image2, test_interval=100)
        # print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')
        dets_centernet, scores_centernet = centernet.detect_image_dets(image2)
        t1 = time.time()
        for _ in range(100):
            dets_centernet, scores_centernet = centernet.detect_image_dets(image2)
        t2 = time.time()
        tact_time = (t2 - t1) / 100
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')
        # print(dets_centernet)
        #print(scores_centernet)