import time
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from ProbEn import ProbEn
from yolov7.yolo_RGB import YOLO_RGB
from yolov7.yolo_T import YOLO_T



if __name__ == '__main__':
    yolo_rgb = YOLO_RGB()
    yolo_T=YOLO_T()
    proben = ProbEn()

    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"

    # -------------------------------------------------------------------------#
    #   dir_origin_RGB_path     指定了用于检测的RGB图片的文件夹路径
    #   dir_origin_T_path      指定了用于检测的红外图片的文件夹路径
    #   注意，必须保证RGB和红外图像的数量一致且图像的名字需完全一一对应
    #
    #   dir_save_path       指定了检测完图片的保存路径
    #   dir_origin_path和dir_save_path在mode='dir_predict'和mode='predict'时均有效
    #
    #   ProbEn融合时在RGB图片上绘制预测框、类别、置信度结果
    # -------------------------------------------------------------------------#
    dir_origin_RGB_path = r"D:\KAIST数据集\重新标注的kaist\kaist_wash_picture_test\visible"
    dir_origin_T_path = r"D:\KAIST数据集\重新标注的kaist\kaist_wash_picture_test\lwir"
    dir_save_path = "img_out/T_new_confi0.3nms0.3/"

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
    video_path = "img/4_Trim.mp4"
    video_save_path = "img_out/4_Trim.mp4"
    video_fps = 25.0

    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/1.jpg"

    # 融合rgb和T的两种模态的图像，融合的ProbEn结果显示在RGB图像上
    # 单张图片预测模式下，需输入RGB图片的路径，且在dir_origin_T_path文件下必须有对应的同名的红外图像
    if mode == 'predict':
        while True:
            img_rgb_path = input('Input RGB_image filename:')
            img_name = os.path.basename(img_rgb_path)

            if os.path.exists(os.path.join(dir_origin_T_path, img_name)):
                img_T_path = os.path.join(dir_origin_T_path, img_name)
            else:
                print(f"{img_name} RGB图像在 {dir_origin_T_path} 文件下无对应的红外图像!")
                break

            try:
                image_rgb = Image.open(img_rgb_path)
                image_T = Image.open(img_T_path)
            except:
                print('Open Error! Try again!')
            else:
                dets_rgb, scores_rgb = yolo_rgb.detect_image_dets(image_rgb)
                dets_rgb = np.asarray(dets_rgb)
                scores_rgb = np.asarray(scores_rgb)

                dets_T, scores_T = yolo_T.detect_image_dets(image_T)
                dets_T = np.asarray(dets_T)
                scores_T = np.asarray(scores_T)

                r_image = proben.fusion_image(image_rgb, dets_rgb, scores_rgb, dets_T, scores_T)
                r_image.show()

    elif mode == 'video':
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测

            # 这一块运用到实际时还需要修改，yolo_rgb和yolo_T调用的参数应该分别是frame_rgb和frame_T
            dets_rgb, scores_rgb = yolo_rgb.detect_image_dets(frame)
            dets_T, scores_T = yolo_T.detect_image_dets(frame)

            if len(dets_rgb) == 0 and len(dets_T) == 0:
                print("无目标")
                frame = np.array(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                fps = (fps + (1. / (time.time() - t1))) / 2
                print("fps= %.2f" % (fps))
                frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("video", frame)
                c = cv2.waitKey(1) & 0xff
                if video_save_path != "":
                    out.write(frame)
                continue
            elif len(dets_T) == 0:
                frame = np.array(yolo_rgb.detect_image(frame))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                fps = (fps + (1. / (time.time() - t1))) / 2
                print("fps= %.2f" % (fps))
                frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("video", frame)
                c = cv2.waitKey(1) & 0xff
                if video_save_path != "":
                    out.write(frame)
                continue
            elif len(dets_rgb) == 0:
                frame = np.array(yolo_T.detect_image(frame))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                fps = (fps + (1. / (time.time() - t1))) / 2
                print("fps= %.2f" % (fps))
                frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("video", frame)
                c = cv2.waitKey(1) & 0xff
                if video_save_path != "":
                    out.write(frame)
                continue
            dets_rgb = np.asarray(dets_rgb)
            scores_rgb = np.asarray(scores_rgb)
            dets_T = np.asarray(dets_T)
            scores_T = np.asarray(scores_T)

            frame = np.array(proben.fusion_image(frame, dets_rgb, scores_rgb, dets_T, scores_T))
            # frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    # 融合rgb和T的两种图像的模态，融合的ProbEn结果显示在RGB图像上
    elif mode == "dir_predict":

        from tqdm import tqdm

        img_RGB_names = os.listdir(dir_origin_RGB_path)
        img_T_names = os.listdir(dir_origin_T_path)

        if len(img_RGB_names) != len(img_T_names):
            raise ValueError("红外图像和RGB图像数量不一致，请检查两者的文件夹！")
        else:
            for img_rgb_name in tqdm(img_RGB_names):
                if img_rgb_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    img_rgb_path = os.path.join(dir_origin_RGB_path, img_rgb_name)

                    if os.path.exists(os.path.join(dir_origin_T_path, img_rgb_name)):
                        img_T_path = os.path.join(dir_origin_T_path, img_rgb_name)
                    else:
                        print(f"{img_rgb_name}RGB图像无对应的红外图像!")
                        break

                    try:
                        image_rgb = Image.open(img_rgb_path)
                        image_T = Image.open(img_T_path)
                    except:
                        print(f"Open Error! Try again!")
                    else:
                        dets_rgb, scores_rgb = yolo_rgb.detect_image_dets(image_rgb)
                        dets_rgb = np.asarray(dets_rgb)
                        scores_rgb = np.asarray(scores_rgb)

                        dets_T, scores_T = yolo_T.detect_image_dets(image_T)
                        dets_T = np.asarray(dets_T)
                        scores_T = np.asarray(scores_T)

                        r_image = proben.fusion_image(image_rgb, dets_rgb, scores_rgb, dets_T, scores_T)

                        if not os.path.exists(dir_save_path):
                            os.makedirs(dir_save_path)
                        r_image.save(os.path.join(dir_save_path, img_rgb_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "fps":
        # ---------------------------------------------------------#
        #   测试fps需要选择两个检测器都能检测到目标的图片,这样才能得到的融合fps
        #   若只有一个检测器检测到时，此时变成单个检测器的fps测试
        # ---------------------------------------------------------#
        image = Image.open(fps_image_path)
        dets_rgb, scores_rgb = yolo_rgb.detect_image_dets(image)
        dets_rgb = np.asarray(dets_rgb)
        scores_rgb = np.asarray(scores_rgb)

        dets_T, scores_T = yolo_T.detect_image_dets(image)
        dets_T = np.asarray(dets_T)
        scores_T = np.asarray(scores_T)

        r_image = proben.fusion_image(image, dets_rgb, scores_rgb, dets_T, scores_T)

        t1 = time.time()
        for _ in range(test_interval):
            dets_rgb, scores_rgb = yolo_rgb.detect_image_dets(image)
            dets_rgb = np.asarray(dets_rgb)
            scores_rgb = np.asarray(scores_rgb)

            dets_T, scores_T = yolo_T.detect_image_dets(image)
            dets_T = np.asarray(dets_T)
            scores_T = np.asarray(scores_T)

            r_image = proben.fusion_image(image, dets_rgb, scores_rgb, dets_T, scores_T)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')