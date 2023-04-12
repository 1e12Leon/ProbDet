import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_FNPI import get_FNPI
from yolo import YOLO

if __name__ == "__main__":
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个FNPI计算流程，包括获得预测结果、获得真实框、计算FNPI。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅获得真实框。
    #   map_mode为3代表仅仅计算FNPI。
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 0
    #--------------------------------------------------------------------------------------#
    #   此处的classes_path用于指定需要测量FNPI的类别
    #   一般情况下与训练和预测所用的classes_path一致即可
    #--------------------------------------------------------------------------------------#
    classes_path    = 'model_data/people_classes_KAIST.txt'
    # classes_path    = 'model_data/voc_classes.txt'
    #--------------------------------------------------------------------------------------#
    #   FNPI_IOU作为判定预测框与真实框相匹配(即真实框所对应的目标被检测成功)的条件
    #   只有大于FNPI_IOU值才算检测成功
    #--------------------------------------------------------------------------------------#
    FNPI_IOU      = 0.5
    #--------------------------------------------------------------------------------------#
    #   confidence的设置与计算map时的设置情况不一样。
    #   受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，因此，map计算时的confidence的值应当设置的尽量小进而获得全部可能的预测框。
    #   而计算FNPI设置的置信度confidence应该与预测时的置信度一致，只有得分大于置信度的预测框会被保留下来
    #--------------------------------------------------------------------------------------#
    confidence      = 0.5
    #--------------------------------------------------------------------------------------#
    #   预测时使用到的非极大抑制值的大小，越大表示非极大抑制越不严格。
    #   该值也应该与预测时设置的nms_iou一致。
    #--------------------------------------------------------------------------------------#
    nms_iou         = 0.3
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    # VOCdevkit_path  = r'E:\pythonProject\object-detection\yolov7-pytorch-master\VOCdevkit'
    VOCdevkit_path  = r'D:\KAIST数据集\重新标注的kaist'
    #-------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out
    #-------------------------------------------------------#
    FNPI_out_path    = 'FNPI_out/FNPI_out_KAIST_RGB'

    # image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()
    image_ids = open(os.path.join(VOCdevkit_path, "kaist_wash_picture_test/test.txt")).read().strip().split()

    if not os.path.exists(FNPI_out_path):
        os.makedirs(FNPI_out_path)
    if not os.path.exists(os.path.join(FNPI_out_path, 'ground-truth')):
        os.makedirs(os.path.join(FNPI_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(FNPI_out_path, 'detection-results')):
        os.makedirs(os.path.join(FNPI_out_path, 'detection-results'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence = confidence, nms_iou = nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            # image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            # image_path  = os.path.join(VOCdevkit_path, "kaist_wash_picture_test/lwir/"+image_id+".jpg")
            image_path  = os.path.join(VOCdevkit_path, "kaist_wash_picture_test/visible/"+image_id+".jpg")
            image       = Image.open(image_path)
            yolo.get_map_txt(image_id, image, class_names, FNPI_out_path)
        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(FNPI_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                # root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
                root = ET.parse(os.path.join(VOCdevkit_path, "kaist_wash_annotation_test/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_FNPI(FNPI_IOU, True, path = FNPI_out_path)
        print("Get map done.")