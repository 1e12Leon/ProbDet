import colorsys
import os
import time

import torch
from PIL import Image, ImageDraw, ImageFont

from scipy.optimize import linear_sum_assignment
import numpy as np
#from numba import jit

from CenterNet.centernet import CenterNet
from yolov7.utils.utils import cvtColor, resize_image, preprocess_input
from yolov7.yolo import YOLO


class ProbEn(object):
    _defaults = {
        # -------------------------------------------------------------------------#
        #   crop                指定了是否在单张图片预测后对目标进行截取
        #   count               指定了是否进行目标的计数
        # -------------------------------------------------------------------------#
        "crop": False,
        "count": False,
        "cuda": True,
    }

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value
        self.voc_classes = ['dog', 'person', 'cat', 'car']
        #KAIST数据集只有person类
        #self.voc_classes = ['person']
        self.num_classes = len(self.voc_classes)
        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    def iou(self, bb_test, bb_gt):
        """
        在两个box间计算IOU
        :param bb_test: box1 = [x1y1x2y2] 即 [左上角的x坐标，左上角的y坐标，右下角的x坐标，右下角的y坐标]
        :param bb_gt: box2 = [x1y1x2y2]
        :return: 交并比IOU
        """
        xx1 = np.maximum(bb_test[0], bb_gt[0])  # 获取交集面积四边形的 左上角的x坐标
        yy1 = np.maximum(bb_test[1], bb_gt[1])  # 获取交集面积四边形的 左上角的y坐标
        xx2 = np.minimum(bb_test[2], bb_gt[2])  # 获取交集面积四边形的 右下角的x坐标
        yy2 = np.minimum(bb_test[3], bb_gt[3])  # 获取交集面积四边形的 右下角的y坐标
        w = np.maximum(0., xx2 - xx1)  # 交集面积四边形的 右下角的x坐标 - 左上角的x坐标 = 交集面积四边形的宽
        h = np.maximum(0., yy2 - yy1)  # 交集面积四边形的 右下角的y坐标 - 左上角的y坐标 = 交集面积四边形的高
        wh = w * h  # 交集面积四边形的宽 * 交集面积四边形的高 = 交集面积
        """
        两者的交集面积，作为分子。
        两者的并集面积作为分母。
        一方box框的面积：(bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        另外一方box框的面积：(bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) 
        """
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
                  + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
                  - wh)
        return o

    """
    利用匈牙利算法对跟踪目标框和yoloV3检测结果框进行关联匹配，整个流程是遍历检测结果框和跟踪目标框，并进行两两的相似度最大的比对。
    相似度最大的认为是同一个目标则匹配成功的将其保留，相似度低的未成功匹配的将其删除。
    使用的是通过yoloV3得到的“并且和预测框相匹配的”检测框来更新卡尔曼滤波器得到的预测框。
        detections：通过yoloV3得到的检测结果框
        trackers：通过卡尔曼滤波器得到的预测结果跟踪目标框
        iou_threshold=0.3：大于IOU阈值则认为是同一个目标则匹配成功将其保留，小于IOU阈值则认为不是同一个目标则未成功匹配将其删除。
        return返回值：
            matches：跟踪成功目标的矩阵。即前后帧都存在的目标，并且匹配成功同时大于iou阈值。
            np.array(unmatched_detections)：新增目标指的就是存在于detections检测结果框当中，但不存在于trackers预测结果跟踪目标框当中。
            np.array(unmatched_trackers)：离开画面的目标指的就是存在于trackers预测结果跟踪目标框当中，但不存在于detections检测结果框当中。
    """

    def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """
        将检测框bbox与卡尔曼滤波器的跟踪框进行关联匹配
        :param detections:通过yolo得到的检测结果框
        :param trackers:通过卡尔曼滤波器得到的预测结果跟踪目标框
        :param iou_threshold:大于IOU阈值则认为是同一个目标则匹配成功将其保留，小于IOU阈值则认为不是同一个目标则未成功匹配将其删除。
        :return:跟踪成功目标的矩阵：matchs。即前后帧都存在的目标，并且匹配成功同时大于iou阈值。
                新增目标的矩阵：unmatched_detections。
                                新增目标指的就是存在于detections检测结果框当中，但不存在于trackers预测结果跟踪目标框当中。
                跟踪失败即离开画面的目标矩阵：unmatched_trackers。
                                离开画面的目标指的就是存在于trackers预测结果跟踪目标框当中，但不存在于detections检测结果框当中。
        """
        """ 
        1.跟踪器链(列表)：
            实际就是多个的卡尔曼滤波KalmanBoxTracker自定义类的实例对象组成的列表。
            每个目标框都有对应的一个卡尔曼滤波器(KalmanBoxTracker实例对象)，
            KalmanBoxTracker类中的实例属性专门负责记录其对应的一个目标框中各种统计参数，
            并且使用类属性负责记录卡尔曼滤波器的创建个数，增加一个目标框就增加一个卡尔曼滤波器(KalmanBoxTracker实例对象)。
            把每个卡尔曼滤波器(KalmanBoxTracker实例对象)都存储到跟踪器链(列表)中。
        2.unmatched_detections(列表)：
            检测框中出现新目标，但此时预测框(跟踪框)中仍不不存在该目标，
            那么就需要在创建新目标对应的预测框/跟踪框(KalmanBoxTracker类的实例对象)，
            然后把新目标对应的KalmanBoxTracker类的实例对象放到跟踪器链(列表)中。
        3.unmatched_trackers(列表)：
            当跟踪目标失败或目标离开了画面时，也即目标从检测框中消失了，就应把目标对应的跟踪框(预测框)从跟踪器链中删除。
            unmatched_trackers列表中保存的正是跟踪失败即离开画面的目标，但该目标对应的预测框/跟踪框(KalmanBoxTracker类的实例对象)
            此时仍然存在于跟踪器链(列表)中，因此就需要把该目标对应的预测框/跟踪框(KalmanBoxTracker类的实例对象)从跟踪器链(列表)中删除出去。
        """
        # 跟踪目标数量为0，直接构造结果
        if (len(trackers) == 0) or (len(detections) == 0):
            """
            如果卡尔曼滤波器得到的预测结果跟踪目标框len(trackers)为0 或者 yoloV3得到的检测结果框len(detections)为0 的话，
            跟踪成功目标的矩阵：matchs 为 np.empty((0, 2), dtype=int)
            新增目标的矩阵：unmatched_detections 为 np.arange(len(detections))
            跟踪失败即离开画面的目标矩阵：unmatched_trackers 为 np.empty((0, 5), dtype=int)
            """
            # return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.arange(len(trackers))

        """ 因为要计算所有检测结果框中每个框 和 所有跟踪目标框中每个框 两两之间 的iou相似度计算，
            即所有检测结果框中每个框 都要和 所有跟踪目标框中每个框 进行两两之间 的iou相似度计算，
            所以iou_matrix需要初始化为len(detections检测结果框) * len(trackers跟踪目标框) 形状的0初始化的矩阵。 """
        # iou 不支持数组计算。逐个计算两两间的交并比，调用 linear_assignment 进行匹配
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        # 遍历目标检测（yoloV3检测）的bbox集合，每个检测框的标识为d，det为检测结果框
        for d, det in enumerate(detections):
            # 遍历跟踪框（卡尔曼滤波器预测）bbox集合，每个跟踪框标识为t，trackers为跟踪目标框
            for t, trk in enumerate(trackers):
                """ 
                遍历每个检测结果框 和 遍历每个跟踪目标框 进行两两之间 的iou相似度计算。
                行索引值对应的是目标检测框。列索引值对应的是跟踪目标框。
                """
                iou_matrix[d, t] = self.iou(det, trk)

        """ 
        row_ind, col_ind=linear_sum_assignment(-iou_matrix矩阵) 
            通过匈牙利算法得到最优匹配度的“跟踪框和检测框之间的”两两组合。
            通过相同下标位置的行索引和列索引即可从iou_matrix矩阵得到“跟踪框和检测框之间的”两两组合最优匹配度的IOU值。
            -iou_matrix矩阵：linear_assignment的输入是cost成本矩阵，IOU越大对应的分配代价应越小，所以iou_matrix矩阵需要取负号。
            row_ind：行索引构建的一维数组。行索引值对应的是目标检测框。
            col_ind：列索引构建的一维数组。列索引值对应的是跟踪目标框。
            比如：
                row_ind：[0 1 2 3]。col_ind列索引：[3 2 1 0]。
                np.array(list(zip(*result)))：[[0 3] [1 2] [2 1] [3 0]]
        """
        iou_matrix = np.asarray(iou_matrix)
        nan = np.isnan(iou_matrix).any()
        if nan:
            iou_matrix[np.isnan(iou_matrix)] = 0

        # 通过匈牙利算法将跟踪框和检测框以[[d,t]...]的二维矩阵的形式存储在match_indices中
        result = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(*result)))

        """ np.array(unmatched_detections)：新增目标指的就是存在于detections检测结果框当中，但不存在于trackers预测结果跟踪目标框当中 """
        # 记录未匹配的检测框及跟踪框
        # 未匹配的检测框放入unmatched_detections中，表示有新的目标进入画面，要新增跟踪器跟踪目标
        unmatched_detections = []
        for d, det in enumerate(detections):
            """ matched_indices[:, 0]：取出的是每行的第一列，代表的是目标检测框。
               如果目标检测框的索引d不存在于匹配成功的matched_indices中每行的第一列的话，代表目标检测框中有新的目标出现在画面中，
               则把未匹配的目标检测框放入到unmatched_detections中表示需要新增跟踪器进行跟踪目标。
            """
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        """ np.array(unmatched_trackers)：离开画面的目标指的就是存在于trackers预测结果跟踪目标框当中，但不存在于detections检测结果框当中 """
        # 未匹配的跟踪框放入unmatched_trackers中，表示目标离开之前的画面，应删除对应的跟踪器
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            """ matched_indices[:, 1]：取出的是每行的第二列，代表的是跟踪目标框。
               如果跟踪目标框的索引t不存在于匹配成功的matched_indices中每行的第二列的话，代表跟踪目标框中有目标离开了画面，
               则把未匹配的跟踪目标框放入到unmatched_trackers中表示需要删除对应的跟踪器。
            """
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        """ matches：跟踪成功目标的矩阵。即前后帧都存在的目标，并且匹配成功同时大于iou阈值。
            即把匹配成功的matched_indices中的并且小于iou阈值的[d,t]放到matches中。
        """
        # 将匹配成功的跟踪框放入matches中
        matches = []
        for m in matched_indices:
            """
            m[0]：每行的第一列，代表的是目标检测框。m[1]：每行的第二列，代表的是跟踪目标框。
            iou_matrix[m[0], m[1]] < iou_threshold：
                根据目标检测框的索引作为行索引，跟踪目标框的索引作为列索引，
                即能找到“跟踪框和检测框之间的”两两组合最优匹配度的IOU值，如果该IOU值小于iou阈值的话，
                则把目标检测框放到unmatched_detections中，把跟踪目标框放到unmatched_trackers中。
            """
            # 过滤掉IOU低的匹配，将其放入到unmatched_detections和unmatched_trackers
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])  # m[0]：每行的第一列，代表的是目标检测框。
                unmatched_trackers.append(m[1])  # m[1]：每行的第二列，代表的是跟踪目标框。
            # 满足条件的以[[d,t]...]的形式放入matches中
            else:
                """ 存储到列表中的每个元素的形状为(1, 2) """
                matches.append(m.reshape(1, 2))

        """
        如果矩阵matches中不存在任何跟踪成功的目标的话，则创建空数组返回。
        numpy.concatenate((a1,a2,...), axis=0)：能够一次完成多个数组a1,a2,...的拼接。
        >>> a=np.array([1,2,3])
        >>> b=np.array([11,22,33])
        >>> c=np.array([44,55,66])
        >>> np.concatenate((a,b,c),axis=0)  # 默认情况下，axis=0可以不写
        array([ 1,  2,  3, 11, 22, 33, 44, 55, 66]) #对于一维数组拼接，axis的值不影响最后的结果
        """
        # 初始化matches,以np.array的形式返回
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            """ 
            np.concatenate(matches, axis=0)：
                [array([[0, 0]], dtype=int64), array([[1, 1]], dtype=int64),  。。。] 转换为 [[0, 0] [1, 1] 。。。]
            """
            matches = np.concatenate(matches, axis=0)  # 默认情况下，axis=0可以不写

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def bayesian_fusion_multiclass2(self, match_score_vec, pred_class, num_cls):
        # ---------------------------------------------------#
        #   match_score_vec:不同检测器置信度列表（所有类别, 应为array形式）
        #   pred_class: 预测结果的索引
        #   num_cls:类别数
        # ---------------------------------------------------#
        scores = np.zeros((match_score_vec.shape[0], num_cls))
        scores[:, :num_cls] = match_score_vec
        # scores[:, -1] = 1 - np.sum(match_score_vec, axis=1)
        # print(scores)
        log_scores = np.log(scores)
        sum_logits = np.sum(log_scores, axis=0)
        exp_logits = np.exp(sum_logits)
        out_score = exp_logits[pred_class] / np.sum(exp_logits)
        return out_score

    def bayesian_fusion_multiclass(self, match_score_vec, pred_class):
        # ---------------------------------------------------#
        #   match_score_vec:不同检测器置信度列表（所有类别, 应为array形式）
        #   pred_class: 预测结果的索引
        #   num_cls:类别数
        # ---------------------------------------------------#
        # scores = np.zeros((match_score_vec.shape[0], 2))
        scores = match_score_vec
        # scores[:, -1] = 1 - np.sum(match_score_vec, axis=1)
        # print(scores)
        log_scores = np.log(scores)
        sum_logits = np.sum(log_scores, axis=0)
        exp_logits = np.exp(sum_logits)
        out_score = exp_logits[pred_class] / np.sum(exp_logits)
        return out_score

    def weighted_box_fusion(self, bbox, score):
        # ---------------------------------------------------#
        #   bbox:  不同检测器的预测框（应为array形式）
        #   score: 不同检测器的置信度（应为array形式）
        # ---------------------------------------------------#
        weight = score / np.sum(score)
        out_bbox = np.zeros(4)
        for i in range(len(score)):
            out_bbox += weight[i] * bbox[i]
        return out_bbox

    def fusion_image(self, image, dets_1, scores_1, dets_2, scores_2):
        # ---------------------------------------------------#
        #   绘制初始化
        # ---------------------------------------------------#
        draw = ImageDraw.Draw(image)
        thickness = int(max((image.size[0] + image.size[1]) // np.mean([640, 640]), 1))  # 厚度
        font = ImageFont.truetype(font=r'D:\Deep_Learning_folds\ProbEn\yolov7\model_data\simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # ---------------------------------------------------#
        #   获取ProbEn输入：
        #   boxs：预测框
        #   scores：置信度
        #   classes：预测类别索引
        # ---------------------------------------------------#
        # ---------------------------------------------------#
        #   dets_1和scores_1为yolo的检测结果和各类的概率
        #   dets的形式：[[top  left  bottom  right  score  预测为第几类的索引],
        #               ...
        #               [top  left  bottom  right  score  预测为第几类的索引]]
        #   scores的形式：[[第一类的概率  第二类的概率  第三类的概率  第n类的概率],
        #               ...
        #                 [第一类的概率  第二类的概率  第三类的概率  第n类的概率]]
        # ---------------------------------------------------#

        if(len(dets_1)==0):
            boxs1 = []
            scores1 = []
            classes1 = []
        else:
            boxs1 = dets_1[:, :4]
            scores1 = dets_1[:, 4]
            classes1 = dets_1[:, 5] # 索引需要转换为int
        if(len(dets_2)==0):
            boxs2 = []
            scores2 = []
            classes2 = []
        else:
            boxs2 = dets_2[:, :4]
            scores2 = dets_2[:, 4]
            classes2 = dets_2[:, 5] # 索引需要转换为int

        # ---------------------------------------------------#
        #   两个检测器的检测结果匹配
        # ---------------------------------------------------#
        matches, unmatched_detection1, unmatched_detection2 = self.associate_detections_to_trackers(boxs1, boxs2)

        # ---------------------------------------------------#
        #   ProbEn融合
        #   只有一个检测器检测到: 直接使用结果，即绘制两个unmatched_detection，无需融合
        #   两个检测器都检测到，但类别不同，需要在matches处理过程中判断，无需融合
        # ---------------------------------------------------#

        dets = []
        # ---------------------------------------------------#
        #   处理二者匹配到的目标(matches)
        # ---------------------------------------------------#
        for index in matches:
            # ----------------------------#
            #   bbox融合
            # ----------------------------#
            bboxs = [boxs1[index[0]], boxs2[index[1]]]
            scores = [scores1[index[0]], scores2[index[1]]]
            classes = [classes1[index[0]], classes2[index[1]]]
            # ----------------------------#
            #   两个检测器类别不同的情况
            # ----------------------------#
            if classes[0] != classes[1]:
                if scores[0] >= scores[1]:
                    out_box = bboxs[0]
                    out_score = scores[0]
                    pred_class = int(classes[0])
                else:
                    out_box = bboxs[1]
                    out_score = scores[1]
                    pred_class = int(classes[1])

                top, left, bottom, right = out_box
                dets.append([top, left, bottom, right, out_score, pred_class])
                # print("dets:", [top, left, bottom, right, out_score, pred_class])
                # ----------------------------#
                #   绘制目标
                # ----------------------------#
                label = '{} {:.2f}'.format(self.voc_classes[pred_class], out_score)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')
                text_origin = np.array([left, top + 1])
                draw.rectangle([left + thickness, top + thickness, right - thickness, bottom - thickness],
                               outline=self.colors[pred_class], width=2)
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[pred_class])
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
                continue

            out_box = self.weighted_box_fusion(bboxs, scores)
            # print(bboxs[0])
            # print(out_box)
            # ----------------------------#
            #   置信度融合
            # ----------------------------#
            scores_vec = [scores_1[index[0]], scores_2[index[1]]]
            # print(classes[0])
            pred_class = int(classes[0])
            # print(scores_vec[0], scores_vec[1])
            out_score = self.bayesian_fusion_multiclass(scores_vec, pred_class)
            # print(out_score)

            top, left, bottom, right = out_box
            dets.append([top, left, bottom, right, out_score, pred_class])
            # print("dets:", [top, left, bottom, right, out_score, pred_class])
            # ----------------------------#
            #   绘制目标
            # ----------------------------#
            label = '{} {:.2f}'.format(self.voc_classes[pred_class], out_score)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            text_origin = np.array([left, top + 1])
            draw.rectangle([left + thickness, top + thickness, right - thickness, bottom - thickness],
                           outline=self.colors[pred_class], width=2)
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[pred_class])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)

        # ---------------------------------------------------#
        #   处理unmatched_detection1
        # ---------------------------------------------------#
        for index1 in unmatched_detection1:
            out_box = boxs1[index1]
            out_score = scores1[index1]
            pred_class = int(classes1[index1])
            top, left, bottom, right = out_box
            dets.append([top, left, bottom, right, out_score, pred_class])
            # print("dets:", [top, left, bottom, right, out_score, pred_class])
            # ----------------------------#
            #   绘制目标
            # ----------------------------#
            label = '{} {:.2f}'.format(self.voc_classes[pred_class], out_score)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            text_origin = np.array([left, top + 1])
            draw.rectangle([left + thickness, top + thickness, right - thickness, bottom - thickness],
                           outline=self.colors[pred_class],
                           width=2)
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[pred_class])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
        # ---------------------------------------------------#
        #   处理unmatched_detection2
        # ---------------------------------------------------#
        for index2 in unmatched_detection2:
            out_box = boxs2[index2]
            out_score = scores2[index2]
            pred_class = int(classes2[index2])
            top, left, bottom, right = out_box
            dets.append([top, left, bottom, right, out_score, pred_class])
            # print("dets:", [top, left, bottom, right, out_score, pred_class])
            # ----------------------------#
            #   绘制目标
            # ----------------------------#
            label = '{} {:.2f}'.format(self.voc_classes[pred_class], out_score)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            text_origin = np.array([left, top + 1])
            draw.rectangle([left + thickness, top + thickness, right - thickness, bottom - thickness],
                           outline=self.colors[pred_class],
                           width=2)
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[pred_class])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)

        return image

    def get_map_txt(self, image_id, class_names, map_out_path, dets_1, scores_1, dets_2, scores_2):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w", encoding='utf-8')
        # ---------------------------------------------------#
        #   获取ProbEn输入：
        #   boxs：预测框
        #   scores：置信度
        #   classes：预测类别索引
        # ---------------------------------------------------#
        if(len(dets_1)==0):
            boxs1 = []
            scores1 = []
            classes1 = []
        else:
            boxs1 = dets_1[:, :4]
            scores1 = dets_1[:, 4]
            classes1 = dets_1[:, 5] # 索引需要转换为int
        if(len(dets_2)==0):
            boxs2 = []
            scores2 = []
            classes2 = []
        else:
            boxs2 = dets_2[:, :4]
            scores2 = dets_2[:, 4]
            classes2 = dets_2[:, 5] # 索引需要转换为int

        # ---------------------------------------------------#
        #   两个检测器的检测结果匹配
        # ---------------------------------------------------#
        matches, unmatched_detection1, unmatched_detection2 = self.associate_detections_to_trackers(boxs1, boxs2)

        # ---------------------------------------------------#
        #   ProbEn融合
        #   只有一个检测器检测到: 直接使用结果，即绘制两个unmatched_detection，无需融合
        #   两个检测器都检测到，但类别不同，需要在matches处理过程中判断，无需融合
        # ---------------------------------------------------#

        dets = []
        # ---------------------------------------------------#
        #   处理二者匹配到的目标(matches)
        # ---------------------------------------------------#
        for index in matches:
            # ----------------------------#
            #   bbox融合
            # ----------------------------#
            bboxs = [boxs1[index[0]], boxs2[index[1]]]
            scores = [scores1[index[0]], scores2[index[1]]]
            classes = [classes1[index[0]], classes2[index[1]]]
            # ----------------------------#
            #   两个检测器类别不同的情况
            # ----------------------------#
            if classes[0] != classes[1]:
                if scores[0] >= scores[1]:
                    out_box = bboxs[0]
                    out_score = scores[0]
                    pred_class = int(classes[0])
                else:
                    out_box = bboxs[1]
                    out_score = scores[1]
                    pred_class = int(classes[1])

                top, left, bottom, right = out_box
                dets.append([top, left, bottom, right, out_score, pred_class])
                # print("dets:", [top, left, bottom, right, out_score, pred_class])
                # print(int(pred_class))
                if self.voc_classes[pred_class] not in class_names:
                    continue
                f.write("%s %s %s %s %s %s\n" % (self.voc_classes[pred_class], out_score,
                                                 str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
                continue

            out_box = self.weighted_box_fusion(bboxs, scores)
            # print(bboxs[0])
            # print(out_box)
            # ----------------------------#
            #   置信度融合
            # ----------------------------#
            scores_vec = [scores_1[index[0]], scores_2[index[1]]]
            # print(classes[0])
            pred_class = int(classes[0])
            # print(scores_vec[0], scores_vec[1])
            out_score = self.bayesian_fusion_multiclass(scores_vec, pred_class)
            # print(out_score)

            top, left, bottom, right = out_box
            dets.append([top, left, bottom, right, out_score, pred_class])
            # print("dets:", [top, left, bottom, right, out_score, pred_class])
            if self.voc_classes[pred_class] not in class_names:
                continue
            f.write("%s %s %s %s %s %s\n" % (self.voc_classes[pred_class], out_score,
                                             str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        # ---------------------------------------------------#
        #   处理unmatched_detection1
        # ---------------------------------------------------#
        for index1 in unmatched_detection1:
            out_box = boxs1[index1]
            out_score = scores1[index1]
            pred_class = int(classes1[index1])
            top, left, bottom, right = out_box
            dets.append([top, left, bottom, right, out_score, pred_class])
            # print("dets:", [top, left, bottom, right, out_score, pred_class])
            if self.voc_classes[pred_class] not in class_names:
                continue
            f.write("%s %s %s %s %s %s\n" % (self.voc_classes[pred_class], out_score,
                                             str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
        # ---------------------------------------------------#
        #   处理unmatched_detection2
        # ---------------------------------------------------#
        # print(boxs1)
        # print(unmatched_detection2)
        for index2 in unmatched_detection2:
            out_box = boxs2[index2]
            out_score = scores2[index2]
            pred_class = int(classes2[index2])
            top, left, bottom, right = out_box
            dets.append([top, left, bottom, right, out_score, pred_class])
            # print("dets:", [top, left, bottom, right, out_score, pred_class])
            if self.voc_classes[pred_class] not in class_names:
                continue
            f.write("%s %s %s %s %s %s\n" % (self.voc_classes[pred_class], out_score,
                                             str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return

    def get_FPS(self, dets_1, scores_1, dets_2, scores_2):
        # ---------------------------------------------------#
        #   获取ProbEn输入：
        #   boxs：预测框
        #   scores：置信度
        #   classes：预测类别索引
        # ---------------------------------------------------#
        if(len(dets_1)==0):
            boxs1 = []
            scores1 = []
            classes1 = []
        else:
            boxs1 = dets_1[:, :4]
            scores1 = dets_1[:, 4]
            classes1 = dets_1[:, 5] # 索引需要转换为int
        if(len(dets_2)==0):
            boxs2 = []
            scores2 = []
            classes2 = []
        else:
            boxs2 = dets_2[:, :4]
            scores2 = dets_2[:, 4]
            classes2 = dets_2[:, 5] # 索引需要转换为int

        # ---------------------------------------------------#
        #   两个检测器的检测结果匹配
        # ---------------------------------------------------#
        matches, unmatched_detection1, unmatched_detection2 = self.associate_detections_to_trackers(boxs1, boxs2)

        # ---------------------------------------------------#
        #   ProbEn融合
        #   只有一个检测器检测到: 直接使用结果，即绘制两个unmatched_detection，无需融合
        #   两个检测器都检测到，但类别不同，需要在matches处理过程中判断，无需融合
        # ---------------------------------------------------#

        dets = []
        # ---------------------------------------------------#
        #   处理二者匹配到的目标(matches)
        # ---------------------------------------------------#
        for index in matches:
            # ----------------------------#
            #   bbox融合
            # ----------------------------#
            bboxs = [boxs1[index[0]], boxs2[index[1]]]
            scores = [scores1[index[0]], scores2[index[1]]]
            classes = [classes1[index[0]], classes2[index[1]]]
            # ----------------------------#
            #   两个检测器类别不同的情况
            # ----------------------------#
            if classes[0] != classes[1]:
                if scores[0] >= scores[1]:
                    out_box = bboxs[0]
                    out_score = scores[0]
                    pred_class = int(classes[0])
                else:
                    out_box = bboxs[1]
                    out_score = scores[1]
                    pred_class = int(classes[1])

                top, left, bottom, right = out_box
                dets.append([top, left, bottom, right, out_score, pred_class])
                # print("dets:", [top, left, bottom, right, out_score, pred_class])
                # print(int(pred_class))
                continue

            out_box = self.weighted_box_fusion(bboxs, scores)
            # print(bboxs[0])
            # print(out_box)
            # ----------------------------#
            #   置信度融合
            # ----------------------------#
            scores_vec = [scores_1[index[0]], scores_2[index[1]]]
            # print(classes[0])
            pred_class = int(classes[0])
            # print(scores_vec[0], scores_vec[1])
            out_score = self.bayesian_fusion_multiclass(scores_vec, pred_class)
            # print(out_score)

            top, left, bottom, right = out_box
            dets.append([top, left, bottom, right, out_score, pred_class])
            # print("dets:", [top, left, bottom, right, out_score, pred_class])


        # ---------------------------------------------------#
        #   处理unmatched_detection1
        # ---------------------------------------------------#
        for index1 in unmatched_detection1:
            out_box = boxs1[index1]
            out_score = scores1[index1]
            pred_class = int(classes1[index1])
            top, left, bottom, right = out_box
            dets.append([top, left, bottom, right, out_score, pred_class])
            # print("dets:", [top, left, bottom, right, out_score, pred_class])

        # ---------------------------------------------------#
        #   处理unmatched_detection2
        # ---------------------------------------------------#
        # print(boxs1)
        # print(unmatched_detection2)
        for index2 in unmatched_detection2:
            out_box = boxs2[index2]
            out_score = scores2[index2]
            pred_class = int(classes2[index2])
            top, left, bottom, right = out_box
            dets.append([top, left, bottom, right, out_score, pred_class])
            # print("dets:", [top, left, bottom, right, out_score, pred_class])

        return