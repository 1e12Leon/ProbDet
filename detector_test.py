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
    img = '1.jpeg'
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
    else:
        print("-----------------")
        print("yolov7:")
        # r_image = yolo.detect_image(image, crop=crop, count=count)
        # r_image.show()
        dets_yolo, scores_yolo = yolo.detect_image_dets(image)
        print(dets_yolo)
        print(scores_yolo)

        """print("-----------------")
        print("centernet:")
        r_image = centernet.detect_image(image, crop = crop, count=count)
        r_image.show()
        dets_centernet = centernet.detect_image_dets(image)
        print(dets_centernet)"""