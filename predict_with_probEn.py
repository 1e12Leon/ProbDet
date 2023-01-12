import colorsys
import time
from PIL import Image, ImageDraw, ImageFont

from scipy.optimize import linear_sum_assignment
import numpy as np
from numba import jit

from CenterNet.centernet import CenterNet
from ProbEn import ProbEn
from yolov7.yolo import YOLO

if __name__ == '__main__':
    centernet = CenterNet()
    yolo = YOLO()
    proben = ProbEn()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    # ----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    # ----------------------------------------------------------------------------------------------------------#
    video_path = "img/1.mp4"
    video_save_path = "2.mp4"
    video_fps = 25.0
    # -------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    if mode == 'predict':
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
        else:
            dets_yolo, scores_yolo = yolo.detect_image_dets(image)
            dets_yolo = np.asarray(dets_yolo)
            scores_yolo = np.asarray(scores_yolo)

            dets_centernet, scores_centernet = centernet.detect_image_dets(image)
            dets_centernet = np.asarray(dets_centernet)
            scores_centernet = np.asarray(scores_centernet)
            r_image = proben.fusion_image(image, dets_yolo, scores_yolo, dets_centernet, scores_centernet)
            r_image.show()

    elif mode == 'video':
        pass

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                dets_yolo, scores_yolo = yolo.detect_image_dets(image)
                dets_yolo = np.asarray(dets_yolo)
                scores_yolo = np.asarray(scores_yolo)

                dets_centernet, scores_centernet = centernet.detect_image_dets(image)
                dets_centernet = np.asarray(dets_centernet)
                scores_centernet = np.asarray(scores_centernet)
                r_image     = proben.fusion_image(image, dets_yolo, scores_yolo, dets_centernet, scores_centernet)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)