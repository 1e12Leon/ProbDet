
import numpy as np
from PIL import Image, ImageDraw, ImageFont



def bayesian_fusion_multiclass(match_score_vec, pred_class, num_cls):
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


def weighted_box_fusion(bbox, score):
    # ---------------------------------------------------#
    #   bbox:  不同检测器的预测框（应为array形式）
    #   score: 不同检测器的置信度（应为array形式）
    # ---------------------------------------------------#
    weight = score / np.sum(score)
    out_bbox = np.zeros(4)
    for i in range(len(score)):
        out_bbox += weight[i] * bbox[i]
    return out_bbox


if __name__ == '__main__':
    # b'car 0.92' 474 255 917 1363
    bboxs = [[474, 255, 917, 1363], [473, 252, 910, 1353]]
    scores = [0.71, 0.61]
    out_box = weighted_box_fusion(np.asarray(bboxs), np.asarray(scores))
    print(out_box)
    # [ 12.97468354  14.46202532 229.02531646 240.51265823]

    match_score_vec = [[0.26, 0.71, 0.03], [0.21, 0.61, 0.18]]
    pred_class = 1
    num_cls = 3
    out_score = bayesian_fusion_multiclass(np.asarray(match_score_vec), pred_class, num_cls)
    print(out_score)
    # 0.9488533245339055

    img1 = 'img/1.jpg'
    image = Image.open(img1)
    draw = ImageDraw.Draw(image)
    thickness = int(max((image.size[0] + image.size[1]) // np.mean([640, 640]), 1))  # 厚度
    # print(thickness)
    text_origin = np.array([255 + 5, 473 + 1])
    font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    draw.rectangle([252 + thickness, 473 + thickness, 1353 - thickness, 910 - thickness], outline='red', width=3)
    draw.text(text_origin, str(0.71), fill=(0, 0, 0), font=font)
    image.save('img/det1.jpg')


    img2 = 'img/1.jpg'
    image = Image.open(img2)
    draw = ImageDraw.Draw(image)
    thickness = int(max((image.size[0] + image.size[1]) // np.mean([640, 640]), 1))  # 厚度
    # print(thickness)
    text_origin = np.array([252 + 5, 474 + 1])
    font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    draw.rectangle([255 + thickness, 474 + thickness, 1363 - thickness, 917 - thickness], outline='green', width=3)
    draw.text(text_origin, str(0.61), fill=(0, 0, 0), font=font)
    image.save('img/det2.jpg')


    img3 = 'img/1.jpg'
    image = Image.open(img3)
    draw = ImageDraw.Draw(image)
    thickness = int(max((image.size[0] + image.size[1]) // np.mean([640, 640]), 1))  # 厚度
    # print(thickness)
    text_origin = np.array([253.61363636 + 5, 473.53787879 + 1])
    font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    draw.rectangle([253.61363636 + thickness, 473.53787879 + thickness, 1358.37878788 - thickness, 913.76515152 - thickness], outline='blue', width=3)
    draw.text(text_origin, str(0.88), fill=(0, 0, 0), font=font)
    image.save('img/fusion.jpg')