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

tfAPIでpredictして、kerasのモデルで分類予測
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

import keras

sys.path.append( r'C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\02_keras_py' )
from predicter import base_predict

# 追加
def predict_class_model(dst, model, img_height, img_width, TTA=None, TTA_rotate_deg=0, TTA_crop_num=0, *args, **kwargs):
    """
    学習済み分類モデルで予測
    引数：
        dst:切り出したarray型の画像
        model:学習済み分類モデル
        img_height, img_width:モデルの入力画像サイズ（modelのデフォルトのサイズである必要あり）
        TTA*:TTA option
        *args, **kwargs:TTA用に引数任意に追加できるようにする。*argsはオプション名なしでタプル型で引数渡せる。**kwargsは辞書型で引数渡せる
    返り値：
        pred[pred_max_id]:確信度
        pred_max_id:予測ラベル
    """
    # 画像のサイズ変更
    x = cv2.resize(dst,(img_height,img_width))
    # 4次元テンソルへ変換
    X = np.expand_dims(x, axis=0)
    # 前処理
    X = X/255.0
    # 予測1画像だけ（複数したい場合は[0]をとる）
    if (TTA == "None") and (TTA_rotate_deg == 0) and (TTA_crop_num == 0):
        pred = model.predict(X)[0]
    else:
        # TTA
        pred = base_predict.predict_tta(model, x, TTA=TTA#'flip'
                                        , TTA_rotate_deg=TTA_rotate_deg
                                        , TTA_crop_num=TTA_crop_num, TTA_crop_size=[(img_height*3)//4, (img_width*3)//4]
                                        , preprocess=1.0/255.0, resize_size=[img_height, img_width])
    #print(pred)
    # 予測確率最大のクラスidを取得
    pred_max_id = np.argmax(pred)#,axis=1)
    return pred[pred_max_id], pred_max_id

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

def predict_image_tfapi(detection_graph, sess
                        , PATH_TO_CKPT, PATH_TO_LABELS, PATH_TO_IMAGE, NUM_CLASSES, output_dir
                        , tfapi_min_score_thresh=1e-5
                        , is_show=True, is_img_save=True
                        , class_model=None, dict_class={0.0:"Car"}, class_min_score_thresh=0.5, model_height=331, model_width=331
                        , TTA=None, TTA_rotate_deg=0, TTA_crop_num=0):
    """
    tfAPIでpredict
    kerasの分類モデル引数にあればそのモデルでもpredict
    Args:
        PATH_TO_CKPT: path to frozen detection graph.pb
        PATH_TO_LABELS: path to labelmap.pbtxt
        PATH_TO_IMAGE: path to image file
        NUM_CLASSES: number of classes
        output: path to output directory of detected image file
        tfapi_min_score_thresh: tfAPIのpredictの閾値
        is_show: 検出画像表示するか
        is_img_save: 検出画像保存するか
        class_model: kerasの分類モデルオブジェクト
        dict_class: kerasの分類モデルのクラスのidとクラス名の辞書型データ
        class_min_score_thresh: kerasの分類モデルのpredictの閾値
        model_height, model_width: 予測するkerasの分類モデルの入力画像サイズ（modelのデフォルトのサイズである必要あり）
        TTA*:TTA option
    """
    # Load the label map.
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    #print(label_map)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    #print(categories)
    category_index = label_map_util.create_category_index(categories)
    #print(category_index)

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
    if class_model is not None:
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
            # 切り出し画像を分類モデルでpredict
            class_conf, class_label_id = predict_class_model(dst, class_model, model_height, model_width, TTA=TTA, TTA_rotate_deg=TTA_rotate_deg, TTA_crop_num=TTA_crop_num)
            #print('class_label_name :', dict_class[class_label_id])
            #print('class_conf :', class_conf)

            # 分類モデルの閾値
            if class_conf >= class_min_score_thresh:
                boxes_class_model.append(box)
                classes_class_id_model.append(int(class_label_id))
                classes_class_name_model.append(str(dict_class[class_label_id]))
                scores_class_model.append(float(class_conf))

        boxes = np.array(boxes_class_model)
        classes = np.array([int(c) for c in classes_class_id_model]) # intでないと vis_util.visualize_boxes_and_labels_on_image_array()でエラーになる
        classes_class_name_model = np.array(classes_class_name_model)
        scores = np.array(scores_class_model)
        # 元の型である4次元テンソルへ変換
        boxes = np.expand_dims(boxes, axis=0)
        classes = np.expand_dims(classes, axis=0)
        classes_class_name_model = np.expand_dims(classes_class_name_model, axis=0)
        scores = np.expand_dims(scores, axis=0)
        print('boxes.shape:', boxes.shape)
        #print(boxes)
        #print(type(classes))
        print('classes.shape:', classes.shape)
        #print(classes)
        #print(type(scores))
        print('scores.shape:', scores.shape)
        #print(scores)
    # ---------------------------------------------------------
    else:
        img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        classes = classes.astype(np.int32) # intでないと vis_util.visualize_boxes_and_labels_on_image_array()でエラーになる

    # 検出数が1のときに検出画像をだすとエラーになるのでtryで囲む
    try:
        # 検出画像保存するか
        if is_img_save == True:
            print("Draw the results of the detection......")
            if boxes.shape == (1, 1, 4):
                vis_boxes = boxes.reshape(1, 4) # 検出数が1つだけのときnp.squeezeするとclassesのshape=()となりエラーになるためreshapeでサイズ固定する
                vis_classes = classes.reshape(1,)
                vis_scores = scores.reshape(1,)
            else:
                vis_boxes = np.squeeze(boxes) # np.squeeze: サイズ1の次元の削除
                vis_classes = np.squeeze(classes)
                vis_scores = np.squeeze(scores)
            #print(vis_boxes.shape)
            #print(vis_classes.shape)
            #print(vis_classes)
            #print(vis_scores.shape)
            # Draw the results of the detection (aka 'visulaize the results')
            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                vis_boxes,
                vis_classes,
                vis_scores,
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                max_boxes_to_draw=int(num[0]), # 描画するbboxの最大数
                min_score_thresh=tfapi_min_score_thresh # 領域予測の閾値
            )
            img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            PIL.Image.fromarray(img_RGB).save(os.path.join(output_dir, str(Path(PATH_TO_IMAGE).stem)+'.jpg')) # ファイル出力

            # 検出画像表示するか
            if is_show == True:
                plt.figure(figsize=(4, 4), dpi = 200)
                plt.imshow(image)
                plt.show()
    except Exception as e:
        print('#### visualize_boxes Exception. img path:', PATH_TO_IMAGE, '#####')
        print(e)

    # ローカル変数の存在をチェック
    if 'classes_class_name_model' in locals():
        # 分類モデルで予測ある時
        return boxes, scores, classes_class_name_model, img_RGB
    else:
        # 分類モデルで予測ない時
        return boxes, scores, classes, img_RGB



def predict_dir_tfapi(detection_graph, sess,
                      PATH_TO_CKPT, PATH_TO_LABELS, PATH_TO_IMAGE_DIR, NUM_CLASSES, output_dir,
                      tfapi_min_score_thresh=1e-5,
                      is_show=True, is_img_save=True,
                      is_overwrite=False,
                      class_model=None, dict_class={0.0:"Car"}, class_min_score_thresh=0.5, model_height=331, model_width=331,
                      TTA=None, TTA_rotate_deg=0, TTA_crop_num=0
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
        # 予測領域1件だけの場合何故かエラーになるからtryで囲む
        try:
            # 出力先に同名ファイルあればpredictしない
            if is_overwrite == False and os.path.isfile(os.path.join(output_dir, os.path.basename(path))):
                continue
            # ---- json用 ----
            #img_name = os.path.basename(path)
            #prediction[img_name] = {}
            # ----------------
            # 画像1件predict
            boxes, scores, classes, img_RGB = predict_image_tfapi(detection_graph, sess
                                                                  , PATH_TO_CKPT, PATH_TO_LABELS, path, NUM_CLASSES, output_dir
                                                                  , tfapi_min_score_thresh=tfapi_min_score_thresh # tfAPIのpredictの閾値
                                                                  , is_show=is_show, is_img_save=is_img_save
                                                                  , class_model=class_model, dict_class=dict_class, class_min_score_thresh=class_min_score_thresh, model_height=model_height, model_width=model_width
                                                                  , TTA=TTA, TTA_rotate_deg=TTA_rotate_deg, TTA_crop_num=TTA_crop_num)
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
            #score_check = {}
            # ----------------
            for box, score, cla in zip(boxes, scores, classes):
                ymin, xmin, ymax, xmax = box
                ymin_list.append(int(im_height*ymin))
                xmin_list.append(int(im_width*xmin))
                ymax_list.append(int(im_height*ymax))
                xmax_list.append(int(im_width*xmax))
                # -------------------------- json用 --------------------------
                #if cla not in score_check:
                #    score_check[cla] = score
                #if cla not in prediction[img_name]:
                #    prediction[img_name][cla]=[]
                #    # JTコンペ用 画像1枚に同じクラス複数無いはずなので同じクラスの結果は追加させない+スコアが一番高いのを採用
                #    if score_check[cla] <= score:
                #        prediction[img_name][cla].append([xmin_list[-1], ymin_list[-1], xmax_list[-1], ymax_list[-1]])
                #        #print(prediction)
                # 画像1枚に同じクラス複数ある場合はこっち
                # prediction[img_name][cla].append([xmin_list[-1], ymin_list[-1], xmax_list[-1], ymax_list[-1]])
                # ------------------------------------------------------------
            # 予測結果データフレーム
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
        except Exception as e:
            print('#### predict Exception. img path:', path, '#####')
            print(e)
            continue

    # 位置情報無いレコードは除く
    df_output_dict_merge = df_output_dict_merge[ (df_output_dict_merge['ymin']!=0.0) & (df_output_dict_merge['xmin']!=0.0) & (df_output_dict_merge['ymax']!=0.0) & (df_output_dict_merge['xmax']!=0.0) ]
    # 位置情報tsvファイル出力
    df_output_dict_merge.to_csv(os.path.join(output_dir, 'output_dict.tsv'), sep='\t', index=False)
    # -------------------------- json用 --------------------------
    #with open(os.path.join(output_dir, 'pred.json'), 'w') as f:
    #    json.dump(prediction, f, indent=4)# インデント付けてjsonファイル出力
    # ------------------------------------------------------------
    return df_output_dict_merge

def out_pred_df_mid_posi(df_output_dict_merge, output_dir):
    """
    kaggleのくずし字コンペ用
    予測したx,yの位置情報をx1,x2,y1,y2の中心位置にしたデータフレーム出力
    """
    image_id = []
    labels = []
    for index, series in tqdm(df_output_dict_merge.iterrows()): # df を1行づつループ回す
        y_mid = (series['ymin'] + series['ymax'])//2
        x_mid = (series['xmin'] + series['xmax'])//2
        labels.append( str(series['detection_classes'])+" "+str(x_mid)+" "+str(y_mid) )
        image_id.append( str(Path(series['image_path']).stem) )
    df = pd.DataFrame({'image_id':image_id, 'labels':labels})

    # image_id一意にする
    image_id = sorted(set(image_id), key=image_id.index)
    #print(image_id)

    # 画像id, ラベル1 x_mid1 y mid1 ラベル2 x_mid2 y mid2 … の形式に変形する
    labels_join = []
    for id in tqdm(image_id):
        labels_list = list(df[df['image_id'] == id]['labels'])
        labels_join_1img = ' '.join(labels_list)
        labels_join.append(labels_join_1img)
    df = pd.DataFrame({'image_id':image_id, 'labels':labels_join})
    #display(df)
    df.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-pb", "--PATH_TO_CKPT", type=str, required=True,
        help="path to frozen detection graph.pb")
    ap.add_argument("-l", "--PATH_TO_LABELS", type=str, required=True,
        help="path to labelmap.pbtxt")
    ap.add_argument("-i", "--PATH_TO_IMAGE", type=str, required=True,
        help="path to image dir")
    ap.add_argument("-n", "--NUM_CLASSES", type=int, default=1,
        help="number of classes")
    ap.add_argument("-is_s", "--is_show", action='store_const', const=True, default=False,
        help="show image")
    ap.add_argument("-is_i", "--is_img_save", action='store_const', const=True, default=False,
        help="save image")
    ap.add_argument("-is_o", "--is_overwrite", action='store_const', const=True, default=False,
        help="出力先に同名ファイルあればpredictしない")
    ap.add_argument("-o", "--output", type=str, required=True,
        help="path to output directory of detected video file")
    ap.add_argument("-tfapi_t", "--tfapi_threshold", type=float, default=0.5,
        help="predict score tfapi_threshold")
    ap.add_argument("-c_model", "--class_model_path", type=str, default="None",
        help="kerasの分類モデルpath")
    ap.add_argument("-c_t", "--class_threshold", type=float, default=0.5,
        help="kerasの分類モデル閾値")
    ap.add_argument("-c_mh", "--model_height", type=int, default=32,
        help="kerasの分類モデルの入力層の縦サイズ")
    ap.add_argument("-c_mw", "--model_width", type=int, default=32,
        help="kerasの分類モデルの入力層の横サイズ")
    ap.add_argument("-dic_class", "--dict_class_tsv", type=str, default="None",
        help="kerasの分類モデルの[クラス名(str型:半角英数), クラスid(int型)]のtsvファイルのpath（headerあり）。例:D:\work\kaggle_kuzushiji-recognition\work\classes\20190816\class_id_master.tsv")
    ap.add_argument("-cust_ob", "--custom_objects", type=str, default="None",
        help="kerasの分類モデルの入力層の横サイズ")
    ap.add_argument("--TTA_flip", type=str, default="None",
        help="kerasの分類モデルのTTA flip")
    ap.add_argument("--TTA_rotate_deg", type=int, default=0,
        help="kerasの分類モデルのTTA rotate deg")
    ap.add_argument("--TTA_crop_num", type=int, default=0,
        help="kerasの分類モデルのTTA crop num")
    args = vars(ap.parse_args())
    #print('is_img_save:', args["is_img_save"])
    print('custom_objects:', args["custom_objects"])
    # keras model load
    if args["class_model_path"] == "None":
        class_model = None
    # OctConv2Dモデルロード用
    elif args["custom_objects"] == "OctConv2D":
        sys.path.append( r'C:\Users\shingo\jupyter_notebook\tfgpu_py36_work\02_keras_py\Git\OctConv-TFKeras' )
        from oct_conv2d import OctConv2D
        class_model = keras.models.load_model(args["class_model_path"]
                                              , custom_objects={'OctConv2D': OctConv2D} # OctConvは独自レイヤーだからcustom_objects の指定必要
                                              , compile=False)
    # custom_objects必要ないモデルロード用
    else:
        class_model = keras.models.load_model(args["class_model_path"], compile=False)

    # kerasの分類モデルのクラスidとクラス名のtsvファイルload
    if args["dict_class_tsv"] == "None":
        dict_class = {}
    else:
        df_class = pd.read_csv(args["dict_class_tsv"], sep='\t')
        dict_class = {}
        for index, series in df_class.iterrows():
            dict_class[series[1]] = series[0]
    #print('dict_class:', dict_class)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args["PATH_TO_CKPT"], 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # 指定ディレクトリの画像全件tfAPI+kerasの分類モデルでpredict
    df_output_dict_merge = predict_dir_tfapi(detection_graph, sess
                                             , args["PATH_TO_CKPT"], args["PATH_TO_LABELS"], args["PATH_TO_IMAGE"], args["NUM_CLASSES"], args["output"]
                                             , tfapi_min_score_thresh=args["tfapi_threshold"]
                                             , is_show=args["is_show"], is_img_save=args["is_img_save"]
                                             , is_overwrite=args["is_overwrite"]
                                             , class_model=class_model
                                             , dict_class=dict_class
                                             , class_min_score_thresh=args["class_threshold"]
                                             , model_height=args["model_height"], model_width=args["model_width"]
                                             , TTA=args["TTA_flip"], TTA_rotate_deg=args["TTA_rotate_deg"], TTA_crop_num=args["TTA_crop_num"]
                                             )

    # kaggleのくずし字コンペ用 予測したx,yの位置情報をx1,x2,y1,y2の中心位置にしたデータフレーム出力
    out_pred_df_mid_posi(df_output_dict_merge, args["output"])

if __name__ == '__main__':
    main()
