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
predictはmetric leraningで実行する
"""
# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

import pandas as pd # 追加
import argparse # 追加
from pathlib import Path # 追加
import time # 追加
import PIL.Image # 追加
import json # 追加
import glob # 追加
from tqdm import tqdm # 追加
import matplotlib.pyplot as plt # 追加

# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")
## PATHにobject_detectionのパス入れてるからいらなそう
##sys.path.append(r'C:\Users\shingo\Git\models\research') # 追加
##sys.path.append(r'C:\\Users\\shingo\\Git\\models\\research\\slim') # 追加
##sys.path.append(r'C:\Users\shingo\Git\models\research\object_detection') # 追加

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# 追加
# ----------------------------- metric leraning ----------------------------- #
def get_l2_sotmax_trained_nasnet_model(output_dir):
    """
    l2_sotmaxのネットワークはLambdaレイヤのせいでモデルの重みであるweight.h5ファイルでしか保存できないためネットワーク構築する
    """
    import os, sys
    sys.path.append(
    r'C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\02_keras_py')
    from model import define_model
    from keras import backend as K
    K.clear_session()

    model, orig_model = define_model.get_fine_tuning_model(output_dir
                                                          , 331, 331, 3
                                                          , 223
                                                          , 'NASNetLarge', 300
                                                          , FCnum=0
                                                          , activation='softmax'
                                                          , skip_bn=True
                                                         )
    # L2 softmax network に変形する
    # softmax関数に通す前にL2ノルムで割って定数倍する
    model = define_model.change_l2_softmax_net(model, alpha=16)
    # compile the model
    optim = define_model.get_optimizers(choice_optim='sgd', lr=0.01, decay=0.0)
    model.compile(loss='categorical_crossentropy'
                  , optimizer=optim
                  , metrics=['acc'])
    model.load_weights(os.path.join(output_dir, 'weights.h5'), by_name=False)
    return model

def model_check(model):
    """ modelのsummary()と各layerをprint """
    print(model.summary())
    for i, layer in enumerate(model.layers):
        print(i, layer.name)

def make_activation_l2_soft_model(load_model):
    """ l2_sotmaxのネットワークの各層のoutputを出力するモデルオブジェクト取得する """
    from keras import models
    # 各レイヤー毎のoutputを取得する
    layer_outputs = []
    for idx, l in enumerate(load_model.layers):
        if idx == len(load_model.layers)-1:
            # l2_sotmaxのネットワークはなぜか出力が複数あるとされてしまう。出力層とるにはmodel.get_output_at(0)必要
            # https://keras.io/getting-started/functional-api-guide/#the-concept-of-layer-node
            layer_outputs.append(load_model.get_output_at(0))
        elif idx >= 1:
            # 入力層以外のoutput
            layer_outputs.append(l.output)
    activation_model = models.Model(inputs=load_model.input, outputs=layer_outputs)
    return activation_model

def make_activation_model(load_model):
    """ 各層のoutputを出力するモデルオブジェクト取得する """
    from keras import models
    # 各レイヤー毎のoutputを取得する
    layers = load_model.layers[1:]
    layer_outputs = [layer.output for layer in layers]
    activation_model = models.Model(inputs=load_model.input, outputs=layer_outputs)
    return activation_model

def predict_metric_from_files(file_list, activation_model, layer_id=1018, target_size=(331, 331)):
    """
    画像パスからpredictして、指定layerでmetricを計算
    Args:
        file_list: 画像パスのリスト
        activation_model: 各層のoutputを取得したモデルオブジェクト
        layer_id: predictしてmetric出すレイヤーのid番号。NasNetLargeなら 1018
        target_size: モデルの入力層のサイズ
    Return:
        指定レイヤーのpredict metricのリスト
    """
    from keras import preprocessing
    # predictして特徴量取得（画像複数件predictできるようにpredict結果はリストで返す）
    metric_list = []
    pbar = tqdm(file_list, file=sys.stdout)
    for f in pbar:
        pbar.set_description("%s" % os.path.basename(f)) # tqdmの進捗バー
        img = preprocessing.image.load_img(f, target_size=target_size)
        x = preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        X = x/255.0
        metric_list.append(activation_model.predict(X)[layer_id]) # layer_id層のpredict output詰める
    return metric_list

def CosSim(img1, img2):
    """
    コサイン距離：ベクトル空間モデルにおいて、文書同士を比較する際に用いられる類似度計算手法。
                  ベクトル同士の成す角度の近さを表現する。1に近ければ類似しており、0に近ければ似ていない
    """
    return np.dot(img1, img2) / (np.linalg.norm(img1) * np.linalg.norm(img2))

def compare_predict_metric(master_metric_list, master_name_list, dst, activation_model, layer_id=1018, target_size=(331, 331)):
    """
    切り取り画像とmaster画像とのpredict metricをコサイン類似度で比較して予測ラベル決める
    Args:
        master_metric_list: マスター画像全件predictしてだしたmetricのリスト
        master_name_list: マスター画像のラベル名リスト。ラベルは文字使わず数値にしておくこと
        dst: 切り出したarray型の画像
        activation_model: 各層のoutputを出力するモデルオブジェクト
        layer_id: dstをpredictしてmetric出すレイヤーのid番号。NasNetLargeなら 1018
        target_size: モデルの入力層のサイズ
    Return:
        切り出した画像のコサイン類似度と予測ラベル名(コサイン類似度が最大になるmaster_name_listの要素)
    """
    x = cv2.resize(dst, target_size)
    x = np.expand_dims(x, axis=0)
    X = x/255.0
    c_metric = activation_model.predict(X)[layer_id] # layer_id層のpredict metric
    cos_sim_list = list(map(lambda m_metric: CosSim(m_metric.flatten(), c_metric.flatten()), master_metric_list)) # ndarray.flatten()で1次元にしないとコサイン類似度計算できない
    top_indices = np.argmax(cos_sim_list) # コサイン類似度が最大のindex
    #print("predict", master_name_list[top_indices], cos_sim_list[top_indices])
    return cos_sim_list[top_indices], int(master_name_list[top_indices])
# --------------------------------------------------------------------------- #

def make_category_index_from_dict_class(dict_class):
    """
    dict_class（kerasのモデルの{class_id:class_name}）から
    category_index（tfAPIのbboxのラベル名辞書。{0: {'id': '1016', 'name': '1016'}…}みたいな形式。bboxに予測クラス書くときにつかう）作成する
    """
    category_index = {}
    for k, v in dict_class.items():
        d = {k: {'id': v, 'name': v}}
        category_index.update(d)
    return category_index

def predict_metric_image_tfapi(PATH_TO_CKPT, PATH_TO_LABELS, PATH_TO_IMAGE, NUM_CLASSES, output_dir,
                        tfapi_min_score_thresh=1e-5, max_boxes_to_draw=300,
                        is_show=True, is_img_save=True,
                        class_activation_model=None, dict_class={0.0:"Car"}, class_min_score_thresh=0.5, model_height=331, model_width=331,
                        master_metric_list=None, master_name_list=None, layer_id=1018,
                        master_metric_list_2=None, class_min_score_thresh_2=0.6, class_unit=6
                        ):
    """
    tfAPIでpredict
    kerasの分類モデル引数にあればそのモデルでもpredict
    modelはmetric learning model
    Args:
        PATH_TO_CKPT: path to frozen detection graph.pb
        PATH_TO_LABELS: path to labelmap.pbtxt
        PATH_TO_IMAGE: path to image file
        NUM_CLASSES: number of classes
        output: path to output directory of detected image file
        tfapi_min_score_thresh: tfAPIのpredictの閾値
        max_boxes_to_draw: 描画するbboxの最大数
        is_show: 検出画像表示するか
        is_img_save: 検出画像保存するか
        class_activation_model: kerasの分類モデルオブジェクト。各層のoutputを出力するモデルオブジェクト
        dict_class: kerasの分類モデルのクラスのidとクラス名の辞書型データ
        class_min_score_thresh: kerasの分類モデルのpredictの閾値
        model_height, model_width: 予測するkerasの分類モデルの入力画像サイズ（modelのデフォルトのサイズである必要あり）
        master_metric_list: マスター画像全件predictしてだしたmetricのリスト
        master_name_list: マスター画像のラベル名リスト。ラベルは文字使わず数値にしておくこと
        layer_id: dstをpredictしてmetric出すレイヤーのid番号。NasNetLargeなら 1018
        master_metric_list_2: 別のマスター画像全件predictしてだしたmetricのリスト
        class_min_score_thresh_2: 別のマスター画像でkerasの分類モデルのpredict再実行するかどうかの閾値
    """
    # Load the label map.
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    #print(label_map)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    #print(categories)
    category_index = label_map_util.create_category_index(categories)
    #print(category_index)

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
    #print(type(classes))
    #print(classes.shape)
    #print(classes)
    #print(type(scores))
    #print(scores.shape)
    #print(scores)
    # -------------------- 分類モデルで予測 --------------------
    if class_activation_model is not None:
        category_index = make_category_index_from_dict_class(dict_class)
        boxes = np.squeeze(boxes) # np.squeeze: サイズ1の次元の削除（4次元テンソルなので3次元にする）
        scores = np.squeeze(scores)
        boxes_class_model = []
        classes_class_id_model = []
        classes_class_name_model = []
        scores_class_model = []
        for box, score in zip(boxes, scores):
            #print(box)
            ymin, xmin, ymax, xmax = box
            #print(ymin, xmin, ymax, xmax)

            # 位置情報無いレコードは除く
            # tfAPIの予測領域は(0,0,0,0)の結果が始まったら後続のレコードもすべて(0,0,0,0)なのでbreakする
            if (ymin == 0.0) & (xmin == 0.0) & (ymax == 0.0) & (xmax == 0.0):
                break

            # 領域予測の閾値
            if score < tfapi_min_score_thresh:
                continue

            img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # (ndarray型の画像データから)検出領域切り出し
            # ndarray型の切り出しは[y:y_max,x:x_max]の順番じゃないとおかしくなる
            # https://qiita.com/tadOne/items/8967f046ca395669329d
            dst = img_RGB[int(img_RGB.shape[0]*ymin):int(img_RGB.shape[0]*ymax)
                                , int(img_RGB.shape[1]*xmin):int(img_RGB.shape[1]*xmax)] # スライス[:]はint型でないとエラー
            # ここで画像表示すると、bbox付き画像保存されない.あくまで確認用
            #print(dst)
            #plt.imshow(dst / 255.)
            #plt.show()
            # 切り出し画像をmetric learning modelでpredict
            class_conf, class_label = compare_predict_metric(master_metric_list, master_name_list, dst, class_activation_model, layer_id=layer_id, target_size=(model_height, model_width))
            #print('master:', class_conf, class_label) # cosとラベル確認用
            ########################### 分類モデルのpredict再実行するかの閾値 ###########################
            if class_conf < class_min_score_thresh_2 and master_metric_list_2 is not None:
                x = cv2.resize(dst, (model_height, model_width))
                x = np.expand_dims(x, axis=0)
                X = x/255.0
                c_metric = class_activation_model.predict(X)[layer_id]

                # trainの一部もマスター画像として、切り出し画像をmetric learning modelでpredict
                cos_sim_mean_list = []
                for i,m_name in enumerate(master_name_list):

                    # trainの一部のマスター画像は1クラス6枚単位なので6枚とる！！！！(class_unit=6)
                    # D:\work\JT_SIGNATE_Contest\01_classes\Images\train_10\test
                    m_metric_list = master_metric_list_2[i*class_unit:(i+1)*class_unit]

                    cos_sim_list = list(map(lambda m_metric: CosSim(m_metric.flatten(), c_metric.flatten()), m_metric_list)) # ndarray.flatten()で1次元にしないとコサイン類似度計算できない
                    cos_sim_mean_list.append(np.mean(cos_sim_list))

                top_id = np.argmax(cos_sim_mean_list) # コサイン類似度が最大のindex
                class_label = master_name_list[top_id]
                class_conf = cos_sim_mean_list[top_id]
                #print('master+train_5:', class_conf, class_label) # cosとラベル確認用
            ##########################################################################################

            # 分類モデルの閾値
            if class_conf >= class_min_score_thresh:
                boxes_class_model.append(box)

                # class_labelに対応するdict_classのidを取得
                for k,v in dict_class.items():
                    if v == str(class_label):
                        classes_class_id_model.append(k)

                classes_class_name_model.append(str(class_label))
                scores_class_model.append(float(class_conf))

        boxes = np.array(boxes_class_model)
        classes = np.array([int(c) for c in classes_class_id_model]) # intでないと vis_util.visualize_boxes_and_labels_on_image_array()でエラーになる
        classes_class_name_model = np.array(classes_class_name_model)
        scores = np.array(scores_class_model)
        # 元の方である4次元テンソルへ変換
        boxes = np.expand_dims(boxes, axis=0)
        classes = np.expand_dims(classes, axis=0)
        classes_class_name_model = np.expand_dims(classes_class_name_model, axis=0)
        scores = np.expand_dims(scores, axis=0)
        #print(type(classes))
        #print(classes.shape)
        #print(classes)
        #print(type(scores))
        #print(scores.shape)
        #print(scores)
    # ---------------------------------------------------------

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes), # np.squeeze: サイズ1の次元の削除
        np.squeeze(classes),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        max_boxes_to_draw=max_boxes_to_draw, # 描画するbboxの最大数
        min_score_thresh=tfapi_min_score_thresh # 領域予測の閾値
    )

    img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 検出画像保存するか
    if is_img_save == True:
        PIL.Image.fromarray(img_RGB).save(os.path.join(output_dir, str(Path(PATH_TO_IMAGE).stem)+'.jpg')) # ファイル出力

    # 検出画像表示するか
    if is_show == True:
        plt.figure(figsize=(4, 4), dpi = 200)
        plt.imshow(image)
        plt.show()

    # ローカル変数の存在をチェック
    if 'classes_class_name_model' in locals():
        # 分類モデルで予測ある時
        return boxes, scores, classes_class_name_model, img_RGB
    else:
        # 分類モデルで予測ない時
        return boxes, scores, classes, img_RGB



def predict_metric_dir_tfapi(PATH_TO_CKPT, PATH_TO_LABELS, PATH_TO_IMAGE_DIR, NUM_CLASSES, output_dir,
                            tfapi_min_score_thresh=1e-5, max_boxes_to_draw=300,
                            is_show=True, is_img_save=True,
                            is_overwrite=False,
                            class_activation_model=None, dict_class={0.0:"Car"}, class_min_score_thresh=0.5, model_height=331, model_width=331,
                            master_metric_list=None, master_name_list=None, layer_id=1018,
                            master_metric_list_2=None, class_min_score_thresh_2=0.6
                            ):
    """
    指定ディレクトリの画像全件tfAPIでpredict
    kerasの分類モデル引数にあればそのモデルでもpredict
    """
    # 予測する画像のパス一覧
    img_path_list = glob.glob(os.path.join(PATH_TO_IMAGE_DIR, "*.*"))
    # 予測結果を保存するフォルダ
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # ---- json用 ----
    prediction = {}
    # ----------------
    # 予測結果一覧データフレーム
    df_output_dict_merge = pd.DataFrame(columns=['image_path', 'ymin','xmin', 'ymax', 'xmax', 'detection_scores', 'detection_classes'])
    # 画像情報1件ずつ取得
    pbar = tqdm(img_path_list, file=sys.stdout)
    for path in pbar:
        pbar.set_description("%s" % os.path.basename(path)) # tqdmの進捗バー
        # 出力先に同名ファイルあればpredictしない
        if is_overwrite == False and os.path.isfile(os.path.join(output_dir, os.path.basename(path))):
            continue
        # ---- json用 ----
        img_name = os.path.basename(path)
        prediction[img_name] = {}
        # ----------------
        # 画像1件predict
        boxes, scores, classes, img_RGB = predict_metric_image_tfapi(PATH_TO_CKPT, PATH_TO_LABELS, path, NUM_CLASSES, output_dir,
                                                                    max_boxes_to_draw=max_boxes_to_draw, # 描画するbboxの最大数
                                                                    tfapi_min_score_thresh=tfapi_min_score_thresh, # tfAPIのpredictの閾値
                                                                    is_show=is_show, is_img_save=is_img_save,
                                                                    class_activation_model=class_activation_model, dict_class=dict_class, class_min_score_thresh=class_min_score_thresh, model_height=model_height, model_width=model_width,
                                                                    master_metric_list=master_metric_list, master_name_list=master_name_list, layer_id=layer_id,
                                                                    master_metric_list_2=master_metric_list_2, class_min_score_thresh_2=class_min_score_thresh_2
                                                                    )
        ymin_list = []
        xmin_list = []
        ymax_list = []
        xmax_list = []
        im_height = img_RGB.shape[0]
        im_width = img_RGB.shape[1]
        boxes = np.squeeze(boxes) # np.squeeze: サイズ1の次元の削除（4次元テンソルなので3次元にする）
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        # ---- JTコンペ json用 ----
        score_check = {}
        # ----------------
        for box, score, cla in zip(boxes, scores, classes):
            ymin, xmin, ymax, xmax = box
            ymin_list.append(int(im_height*ymin))
            xmin_list.append(int(im_width*xmin))
            ymax_list.append(int(im_height*ymax))
            xmax_list.append(int(im_width*xmax))
            # -------------------------- json用 --------------------------
            if cla not in score_check:
                score_check[cla] = score
            if cla not in prediction[img_name]:
                prediction[img_name][cla]=[]
                # JTコンペ用 画像1枚に同じクラス複数無いはずなので同じクラスの結果は追加させない+スコアが一番高いのを採用
                if score_check[cla] <= score:
                    prediction[img_name][cla].append([xmin_list[-1], ymin_list[-1], xmax_list[-1], ymax_list[-1]])
                    #print(prediction)
            # 画像1枚に同じクラス複数ある場合はこっち
            # prediction[img_name][cla].append([xmin_list[-1], ymin_list[-1], xmax_list[-1], ymax_list[-1]])
            # ------------------------------------------------------------
        # 予測結果データフレーム これには同じクラス複数残す
        df_output_dict = pd.DataFrame({'image_path': [path]*len(boxes),
                                        'ymin': ymin_list,
                                        'xmin': xmin_list,
                                        'ymax': ymax_list,
                                        'xmax': xmax_list,
                                        'detection_scores': list(scores),
                                        'detection_classes': list(classes)})
        df_output_dict_merge = pd.concat([df_output_dict_merge, df_output_dict])
        #print(df_output_dict_merge)
        #break # 1件だけ確認用

    # 位置情報無いレコードは除く
    df_output_dict_merge = df_output_dict_merge[ (df_output_dict_merge['ymin']!=0.0) & (df_output_dict_merge['xmin']!=0.0) & (df_output_dict_merge['ymax']!=0.0) & (df_output_dict_merge['xmax']!=0.0) ]
    # 位置情報tsvファイル出力
    df_output_dict_merge.to_csv(os.path.join(output_dir, 'output_dict.tsv'), sep='\t', index=False)
    # -------------------------- json用 --------------------------
    with open(os.path.join(output_dir, 'pred.json'), 'w') as f:
        json.dump(prediction, f, indent=4)# インデント付けてjsonファイル出力
    # ------------------------------------------------------------
    return
