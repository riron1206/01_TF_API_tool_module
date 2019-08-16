"""
以下のgithubのコード
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/blob/master/README.md

######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description:
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

tfAPIで画像1枚だけpredictして保存
"""
#Usage:
# cd C:\Users\shingo\Git\TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10\work_20190402
# python ../../Object_detection_image_edit.py -pb inference_graph\frozen_inference_graph.pb -l ../../label_map.pbtxt -i ../../../images/test/cam_image2.jpg -n 6 -s True -o predict_result/image

# Import packages
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import sys

import argparse # 追加
from pathlib import Path # 追加
import time # 追加

# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")
## PATHにobject_detectionのパス入れてるからいらなそう
##sys.path.append(r'C:\Users\shingo\Git\models\research') # 追加
##sys.path.append(r'C:\\Users\\shingo\\Git\\models\\research\\slim') # 追加
##sys.path.append(r'C:\Users\shingo\Git\models\research\object_detection') # 追加

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

## Name of the directory containing the object detection module we're using
#MODEL_NAME = 'inference_graph'
#IMAGE_NAME = 'test1.jpg'
#
## Grab path to current working directory
#CWD_PATH = os.getcwd()
#
## Path to frozen detection graph .pb file, which contains the model that is used
## for object detection.
#PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
#
## Path to label map file
#PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
#
## Path to image
#PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
#
## Number of classes the object detector can identify
#NUM_CLASSES = 6

# 追加
ap = argparse.ArgumentParser()
ap.add_argument("-pb", "--PATH_TO_CKPT", type=str, required=True,
    help="path to frozen detection graph.pb")
ap.add_argument("-l", "--PATH_TO_LABELS", type=str, required=True,
    help="path to labelmap.pbtxt")
ap.add_argument("-i", "--PATH_TO_IMAGE", type=str, required=True,
    help="path to image file")
ap.add_argument("-n", "--NUM_CLASSES", type=int, default=1,
    help="number of classes")
ap.add_argument("-s", "--is_show", type=bool, default=False,
    help="show image")
ap.add_argument("-o", "--output", type=str, required=True,
    help="path to output directory of detected video file")
ap.add_argument("-t", "--threshold", type=float, default=0.5,
    help="predict score threshold")
args = vars(ap.parse_args())

PATH_TO_CKPT = args["PATH_TO_CKPT"]
PATH_TO_LABELS = args["PATH_TO_LABELS"]
PATH_TO_IMAGE = args["PATH_TO_IMAGE"]
NUM_CLASSES = args["NUM_CLASSES"]
is_show = args["is_show"]
output_dir = args["output"]
os.makedirs(output_dir, exist_ok=True)

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
image = cv2.imread(PATH_TO_IMAGE)
image_expanded = np.expand_dims(image, axis=0)

# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# 1画像のpredict全件データフレームにしてtsvで保存
boxes = np.squeeze(boxes) # np.squeeze: サイズ1の次元の削除（4次元テンソルなので3次元にする）
print(boxes[:,0].shape, np.squeeze(scores).shape, np.squeeze(classes).shape, num)
img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
ymin = boxes[:,0]*img_RGB.shape[0] # y,xの位置情報を元画像のサイズに合わせる
xmin = boxes[:,1]*img_RGB.shape[1]
ymax = boxes[:,2]*img_RGB.shape[0]
xmax = boxes[:,3]*img_RGB.shape[1]
df_pred = pd.DataFrame({'ymin':ymin.astype('int64')
                        , 'xmin':xmin.astype('int64')
                        , 'ymax':ymax.astype('int64')
                        , 'xmax':xmax.astype('int64')
                        , 'score':np.squeeze(scores)
                        , 'classes':np.squeeze(classes)
                       })
df_pred.to_csv(os.path.join(output_dir, str(Path(PATH_TO_IMAGE).stem)+'.tsv'), sep='\t')
#print(boxes, scores, classes, num)

# Draw the results of the detection (aka 'visulaize the results')

vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    max_boxes_to_draw=int(num[0]), # 描画するbboxの最大数
    min_score_thresh=args["threshold"])

# 保存
import PIL.Image
img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
PIL.Image.fromarray(img_RGB).save(os.path.join(output_dir, str(Path(PATH_TO_IMAGE).stem)+'.jpg')) # ファイル出力

# opencvで画像表示させるか
if is_show == True:
    # All the results have been drawn on image. Now display the image.
    cv2.imshow('Object detector', image)
    # Press any key to close the image
    cv2.waitKey(0)
    # 3秒待つ
    #time.sleep(3)
    # Clean up
    cv2.destroyAllWindows()
