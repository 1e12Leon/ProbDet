import glob
import json
import operator
import os
import shutil
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

"""
 throw error and exit
"""
def error(msg):
    print(msg)
    sys.exit(0)

"""
 Convert the lines of a file to a list
"""
def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

"""
 Plot - adjust axes
"""
def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])

"""
 Draw plot using Matplotlib
"""
def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar != "":
        """
         Special case to draw in:
            - green -> TP: True Positives (object detected and matches ground-truth)
            - red -> FP: False Positives (object detected but does not match ground-truth)
            - orange -> FN: False Negatives (object not detected but present in the ground-truth)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val) # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15 # in percentage of the figure height
    bottom_margin = 0.05 # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()

def get_FNPI(FNPI_IOU, draw_plot, path):
    GT_PATH = os.path.join(path, 'ground-truth')
    DR_PATH = os.path.join(path, 'detection-results')
    TEMP_FILES_PATH = os.path.join(path, '.temp_files')
    RESULTS_FILES_PATH = os.path.join(path, 'results')

    if not os.path.exists(TEMP_FILES_PATH):
        os.makedirs(TEMP_FILES_PATH)

    if os.path.exists(RESULTS_FILES_PATH):
        shutil.rmtree(RESULTS_FILES_PATH)
    else:
        os.makedirs(RESULTS_FILES_PATH)

    TEMP_FILES_DETECTIONS_PATH = os.path.join(TEMP_FILES_PATH,'classes_detections')
    TEMP_FILES_GT_PATH = os.path.join(TEMP_FILES_PATH,'ground-truth')
    if not os.path.exists(TEMP_FILES_DETECTIONS_PATH):
        os.makedirs(TEMP_FILES_DETECTIONS_PATH)
    if not os.path.exists(TEMP_FILES_GT_PATH):
        os.makedirs(TEMP_FILES_GT_PATH)

    if draw_plot:
        try:
            matplotlib.use('TkAgg')
        except:
            pass

    #   ground_truth_files_list中具有map_out/ground-truth中的所有真实框文件的路径,例如'map_out_RGB_new\ground-truth\set06_V004_l01853.txt'
    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    gt_counter_per_class = {}  # 每一个类有多少真实框
    counter_images_per_class = {}  # 每一个类有多少张图片

    #   对每一个ground_truth_files_list中的真实框文件的路径
    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        #  用GT_PATH文件夹下真实框txt文件的basename找到DR_PATH文件夹下的预测框txt文件
        temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error(error_msg)
        #   拿到真实框文件中所有真实框信息，一条或多条(列表)
        lines_list = file_lines_to_list(txt_file)
        bounding_boxes = []  # 经split()处理后拿到真实框文件中所有真实框信息，一条或多条(列表)
        is_difficult = False
        already_seen_classes = []  # 存放这一个真实框txt文件中有哪些类的类名
        #   对每一条真实框信息
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
                else:
                    class_name, left, top, right, bottom = line.split()
            except:
                if "difficult" in line:
                    line_split = line.split()
                    _difficult = line_split[-1]
                    bottom = line_split[-2]
                    right = line_split[-3]
                    top = line_split[-4]
                    left = line_split[-5]
                    class_name = ""
                    for name in line_split[:-5]:
                        class_name += name + " "
                    class_name = class_name[:-1]
                    is_difficult = True
                else:
                    line_split = line.split()
                    bottom = line_split[-1]
                    right = line_split[-2]
                    top = line_split[-3]
                    left = line_split[-4]
                    class_name = ""
                    for name in line_split[:-4]:
                        class_name += name + " "
                    class_name = class_name[:-1]

            bbox = left + " " + top + " " + right + " " + bottom
            if is_difficult:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

        with open(TEMP_FILES_GT_PATH + "/" + file_id + ".json", 'w') as outfile:
            json.dump(bounding_boxes,
                      outfile)  # 将bounding_boxes的真实框信息存放在TEMP_FILES_GT_PATH文件夹下的命名为 file_id的json文件里，file_id为真实框txt文件的basename

    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)  # 根据gt_counter_per_class字典拿到包含有真实框类别名字的gt_classes列表及n_classes类别数

    dr_files_list = glob.glob(
        DR_PATH + '/*.txt')  # dr_files_list中具有map_out/detection-results中的所有预测框文件的路径,例如'map_out_RGB_new\detection-results\set06_V004_l01853.txt'
    dr_files_list.sort()
    #   对包含有真实框类别名字的gt_classes列表循环
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []  # #   经split()处理后拿到预测框所属类别名与真实框类别名一致中所有此类的预测框信息，形式为：confidence,file_id,bbox
        for txt_file in dr_files_list:  # 对每一个预测框txt文件进行循环
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_PATH, (file_id + ".txt"))  # 用DR_PATH文件夹下真实框txt文件的basename找到GT_PATH文件夹下的预测框txt文件
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:  # 对每一个预测框文件中的每一行（每一张预测图片中的每一条预测结果）进行循环
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except:
                    line_split = line.split()
                    bottom = line_split[-1]
                    right = line_split[-2]
                    top = line_split[-3]
                    left = line_split[-4]
                    confidence = line_split[-5]
                    tmp_class_name = ""
                    for name in line_split[:-5]:
                        tmp_class_name += name + " "
                    tmp_class_name = tmp_class_name[:-1]

                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox, "used": False})

        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)  # 对bounding_boxes上的所有此类的预测框框按置信度降序排序
        with open(TEMP_FILES_DETECTIONS_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)  # 将bounding_boxes中所有此类的预测框信息存放在TEMP_FILES_DETECTIONS_PATH文件夹下的命名为 class_name_dr的json文件里

    #   经过以上两个大for循环的处理得到.temp_files文件夹下的json文件（有两大类）

    sum_FNPI = 0.0  # 各类的漏检率之和
    fnpi_dictionary = {}  # 记录各类的漏检率的字典
    with open(RESULTS_FILES_PATH + "/results.txt", 'w') as results_file:
        results_file.write("# FNPI per class\n")

        #   对真实标签中的每一类循环
        for class_index, class_name in enumerate(gt_classes):
            dr_file = TEMP_FILES_DETECTIONS_PATH + "/" + class_name + "_dr.json"  # 此类预测框json文件（.temp_files/classes_detections文件夹下）
            dr_data = json.load(open(dr_file))
            ground_truth_jsons_list = glob.glob(TEMP_FILES_GT_PATH + '/*.json')

            nd = counter_images_per_class[class_name]
            fnpi_thisClass_per_image = [0] * nd  # 含有此类的图片的漏检率数组
            idx = 0  # 作为fnpi_thisClass_per_image数组中的索引以此记录含有此类的图片的漏检率

            sum_allImage_fnpi_thisClass = 0  # 此类所有图的漏检率之和
            fnpi_thisClass = 0  # 此类的漏检率

            #   对.temp_files文件夹下的每一个真实框json文件(相当于每一张图片)
            for ground_truth_json in ground_truth_jsons_list:
                jsonFile_id = os.path.basename(os.path.normpath(ground_truth_json))
                jsonFile_id = jsonFile_id.split(".json", 1)[0]
                ground_truth_json_data = json.load(open(ground_truth_json))
                if len(ground_truth_json_data) == 0:
                    continue
                gt_thisClass_count_image = 0  # 此真实框json文件中含有此类真实框的个数
                dt_success_thisClass_count_image = 0  # 检测成功的预测框的个数
                fnpi_thisClass_image = 0  # 此类中此图的漏检率

                #   拿到此类别预测框json文件中与真实框json文件File_id一致的所有预测框形成数组
                detection_fileid_in_json = [detection for i,detection in enumerate(dr_data) if detection["file_id"] == jsonFile_id]

                #    对真实框json文件中的每一个真实框
                for ground_truth in ground_truth_json_data:

                    #   真实框类名与此类一致且不是识别困难的目标
                    if ground_truth["class_name"] == class_name and "difficult" not in ground_truth:
                        ovmax = -1
                        gt_match = -1
                        gt_thisClass_count_image += 1
                        bbgt = [float(x) for x in ground_truth["bbox"].split()]

                        #   对上面拿到的预测框数组进行遍历(即对每一个符合要求的预测框)
                        for detection in detection_fileid_in_json:
                            if detection["used"] != True:  # 如果此检测框未标记为使用过(即并未有满足阀值的iou最大的真实框相匹配检测成功)
                                bb = [float(x) for x in detection["bbox"].split()]
                                bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]),
                                      min(bb[3], bbgt[3])]
                                iw = bi[2] - bi[0] + 1
                                ih = bi[3] - bi[1] + 1
                                if iw > 0 and ih > 0:  # 计算IOU
                                    ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                                      + 1) * (
                                                 bbgt[3] - bbgt[1] + 1) - iw * ih
                                    ov = iw * ih / ua
                                    if ov > ovmax:
                                        ovmax = ov
                                        gt_match = detection

                        min_overlap = FNPI_IOU

                        if ovmax >= min_overlap:
                            if not bool(gt_match["used"]):
                                dt_success_thisClass_count_image += 1
                                gt_match[
                                    "used"] = True  # 用于标记此预测框detection已经有符合FNPI_IOU阀值的且iou最大的真实框相对应，相当于这一个真实框成功有预测框检测到

                if gt_thisClass_count_image != 0:
                    fnpi_thisClass_image = 1 - (dt_success_thisClass_count_image / gt_thisClass_count_image)
                    sum_allImage_fnpi_thisClass += fnpi_thisClass_image
                    fnpi_thisClass_per_image[idx] = fnpi_thisClass_image
                    idx += 1

            fnpi_thisClass = sum_allImage_fnpi_thisClass / counter_images_per_class[class_name]
            fnpi_dictionary[class_name] = fnpi_thisClass
            sum_FNPI += fnpi_thisClass

            #   写results.txt文件前面部分
            text = "{0:.2f}%".format(fnpi_thisClass * 100) + " = " + class_name + " FNPI "
            rounded_fnpi_thisClass_per_image = ['%.2f' % elem for elem in fnpi_thisClass_per_image]
            results_file.write(
                text + "\n" + class_name + "_fnpi_per_image: " + str(rounded_fnpi_thisClass_per_image) + "\n\n")

        if n_classes == 0:
            print("未检测到任何种类，请检查标签信息与get_FNPI.py中的classes_path是否修改。")
            return 0
        results_file.write("\n# mFNPI of all classes\n")
        mFNPI = sum_FNPI / n_classes
        text = "mFNPI = {0:.2f}%".format(mFNPI * 100)
        results_file.write(text + "\n")
        print(text)

    # shutil.rmtree(TEMP_FILES_PATH)


    """
    Write number of ground-truth objects per class to results.txt
    """
    with open(RESULTS_FILES_PATH + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of ground-truth objects per class\n")
        for class_name in sorted(gt_counter_per_class):
            results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

    """
    Write number of images per class to results.txt
    """
    with open(RESULTS_FILES_PATH + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of images per class\n")
        for class_name in sorted(counter_images_per_class):
            results_file.write(class_name + ": " + str(counter_images_per_class[class_name]) + "\n")

    """
    Plot the total number of images of each class 
    """
    if draw_plot:
        window_title = "counter-images-per-class-info"
        plot_title = "counter-images-per-class\n"
        plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
        x_label = "counter-images-per-class"
        output_path = RESULTS_FILES_PATH + "/counter-images-per-class-info.png"
        to_show = False
        plot_color = 'forestgreen'
        draw_plot_func(
            counter_images_per_class,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            '',
        )

    """
    Draw mFNPI plot (Show FNPI's of all classes in decreasing order)
    """
    if draw_plot:
        window_title = "mFNPI"
        plot_title = "mFNPI = {0:.2f}%".format(mFNPI*100)
        x_label = "Average FNPI"
        output_path = RESULTS_FILES_PATH + "/mFNPI.png"
        to_show = True
        plot_color = 'royalblue'
        draw_plot_func(
            fnpi_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
        )
    return mFNPI

