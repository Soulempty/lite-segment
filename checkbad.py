import os
import numpy as np
from tools import SegmentationMetric


if __name__ == '__main__':
    pred_label_path = 'work_dirs/checkpoint/steel/results' 
    label_path = '../../dataset/steel/labels/val'
    img_path = '../../dataset/steel/images/val' 
    class_names = ["cls_1","cls_2","cls_3","cls_4"]
    save_path = 'work_dirs/checkpoint/steel/check_bad'
    metric = SegmentationMetric(pred_label_path,label_path,img_path,save_path,class_names)
    metric.compute_metrics()



