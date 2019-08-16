"""
tfAPIのtfrecord作るために必要な物体の位置書いたアノテーションcsvファイル準備するためのコード
"""

import os
import numpy as np
import json
import shutil
from PIL import Image

def make_xywh_tfAPI_csv(anno_df, img_dir, xywh_csv_path='xywh_train.csv', is_class_one=False):
    """
    Object-Detection-API 用に
    (x1, y1, x2, y2, label_id, file_name)の列のデータフレームから
    正解の座標列（'filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'）のcsvファイルを作成する
    Args:
        anno_df: (x1, y1, x2, y2, label_id, file_name)の列のデータフレーム
        img_dir: file_name列の画像ディレクトリのパス
        xywh_csv_path: 出力するcsvファイルパス
        is_class_one: クラスラベルを1で固定する（物体検出にする）場合はTrueにするフラグ
    Return:
        (x1, y1, x2, y2, label_id, file_name)の列のデータフレーム
        (この関数内でcsvファイルに保存するデータ)
    """
    anno_df = anno_df.rename(columns={'file_name': 'filename'
                                      , 'label_id': 'class'
                                      , 'x1': 'xmin'
                                      , 'y1': 'ymin'
                                      , 'x2': 'xmax'
                                      , 'y2': 'ymax'})
    print('anno_df\n', anno_df.head())

    # 画像の大きさ取得
    width_list = []
    height_list = []
    file_names = anno_df['filename']
    for f in file_names:
        path = os.path.join(img_dir, f)
        if os.path.isfile(path): # ファイルの存在を確認
            img = Image.open(path) # PILで画像ファイルロード
            width, height = img.size
            width_list.append(width)
            height_list.append(height)
        else: # ファイルなければ0にしておく
            width_list.append(0)
            height_list.append(0)
    anno_df['width'] = width_list
    anno_df['height'] = height_list

    # 1クラスだけにするか
    if is_class_one == True:
        anno_df['class'] = '1'

    # 列の順番変える
    anno_df = anno_df.loc[:,['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']]
    anno_df['xmin'] = anno_df['xmin'].astype(np.int64)
    anno_df['ymin'] = anno_df['ymin'].astype(np.int64)
    anno_df['xmax'] = anno_df['xmax'].astype(np.int64)
    anno_df['ymax'] = anno_df['ymax'].astype(np.int64)
    anno_df['width'] = anno_df['width'].astype(np.int64)
    anno_df['height'] = anno_df['height'].astype(np.int64)

    # width列が0のレコード削除
    anno_df = anno_df[anno_df['width']!=0]
    print('\nanno_df.shape', anno_df.shape)

    # index 振り直し
    anno_df = anno_df.reset_index(drop=True)
    # csv保存
    anno_df.to_csv(xywh_csv_path, sep=',', header=True, index=False)
    print(xywh_csv_path+'\n', anno_df.head())

    return anno_df
