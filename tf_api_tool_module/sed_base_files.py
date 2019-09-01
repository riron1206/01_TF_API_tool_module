"""
tfAPIの基本ファイル(02_make_tf_file.batとか)のファイルパスを置換する
"""
import os, sys
import argparse
import glob
import pathlib

def sed(file_path, before_str, after_str, out_path=None):
    """
    python でsed コマンド再現。指定ファイルの指定文字を置換して別名ファイル出力。
    https://qiita.com/tsukapah/items/f2235a38592a7082662b
    https://pythonmemo.hatenablog.jp/entry/2018/05/05/135955
    Arges:
        file_path: 置換するファイルパス
        before_str: 置換前の文字
        after_str: 置換後の文字
        out_path: 出力ファイルパス。Noneなら_sed をつけたファイル名で出力
    """
    import sys
    import re
    # ファイルロード
    with open(file_path, mode='r', encoding="utf-8") as f:
        body = f.read()
    # re.sub()で置換 flags=re.DOTALLを指定したことで、正規表現の.に改行コード\nも含まれるようになり、複数行が置換対象
    sed_body = re.sub(before_str, after_str, body, flags=re.DOTALL)
    # ファイル出力
    with open(out_path, mode='w', encoding="utf-8") as f:
        f.write(sed_body)
    #print('out_path:', out_path)
    return

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-P", "--PATH_TO_BE_CONFIGURED", type=str, required=True,
        help="学習の出力先ディレクトリパス。必ず「/」区切りにすること (例: D:/work/kaggle_kuzushiji-recognition/work/object_detection/20190812)")
    ap.add_argument("-I", "--IMAGE_DIR", type=str, required=True,
        help="train setの画像ディレクトリ。tfファイル作成(02_make_tf_file.bat)で使うので「\」区切りでも大丈夫 (例: D:\work\kaggle_kuzushiji-recognition\OrigData\kuzushiji-recognition\train_images)")
    ap.add_argument("-note", "--notebook", type=str, default="01_Object-Detection-API.ipynb",
        help="workdir\01_Object-Detection-API.ipynb")
    ap.add_argument("-02_bat", "--02_make_tf_file", type=str, default="02_make_tf_file.bat",
        help="workdir\02_make_tf_file.bat")
    ap.add_argument("-03_bat", "--03_sed_config_file", type=str, default="03_sed_config_file.bat",
        help="workdir\03_sed_config_file.bat")
    ap.add_argument("-04_train_bat", "--04_1_run_train", type=str, default="04_1_run_train.bat",
        help="workdir\04_1_run_train.bat")
    ap.add_argument("-04_tensorboard_bat", "--04_2_run_tensorboard", type=str, default="04_2_run_tensorboard.bat",
        help="workdir\04_2_run_tensorboard.bat")
    ap.add_argument("-05_bat", "--05_deploy_trained_model", type=str, default="05_deploy_trained_model.bat",
        help="workdir\05_deploy_trained_model.bat")
    ap.add_argument("-06_bat", "--06_predict_1img_test", type=str, default="06_predict_1img_test.bat",
        help="workdir\06_predict_1img_test.bat")
    ap.add_argument("-06_video_bat", "--06_predict_1video_test", type=str, default="06_predict_1video_test.bat",
        help="workdir\06_predict_1video_test.bat")
    ap.add_argument("-07_bat", "--07_predict_keras", type=str, default="07_predict_keras.bat",
        help="workdir\07_predict_kerasbat")
    ap.add_argument("-K", "--PATH_KERAS_MODEL", type=str, default="D:/work/kaggle_kuzushiji-recognition/work/classes/20190816/best_val_acc.h5",
        help="予測で使うkerasの画像分類モデルのパス")
    ap.add_argument("-K_H", "--KERAS_MODEL_HEIGHT", type=int, default=32,
        help="予測で使うkerasの画像分類モデルの入力層の大きさ")
    ap.add_argument("-K_W", "--KERAS_MODEL_WIDTH", type=int, default=32,
        help="予測で使うkerasの画像分類モデルの入力層の大きさ")
    ap.add_argument("-K_CLA", "--PATH_KERAS_DICT_CLASS_TSV", type=str, default="D:/work/kaggle_kuzushiji-recognition/work/classes/20190816/tfAPI_dict_class.tsv",
        help="予測で使うkerasの画像分類モデルのクラス名とクラスidのtsvファイル")
    ap.add_argument("-K_CUS", "--KERAS_CUSTOM_OBJECTS", type=str, default="None",
        help="予測で使うkerasの画像分類モデルロード時に必要なcustom_objects。OctConv2Dとか。デフォルトのNoneならなしにする")
    ap.add_argument("-P_I_D", "--PRED_IMG_DIR", type=str, default="D:/work/kaggle_kuzushiji-recognition/GrayImg/test_images",
        help="予測する画像ディレクトリ")
    args = vars(ap.parse_args())

    # 01_Object-Detection-API.ipynb コピーして置換
    sed(r'C:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\tf_api_train_base_files\01_Object-Detection-API.ipynb'
        , 'PATH_TO_BE_CONFIGURED'
        , args['PATH_TO_BE_CONFIGURED']
        , out_path=args['notebook'])
    sed(args['notebook'], 'IMAGE_DIR', args['IMAGE_DIR'], out_path=args['notebook'])

    # 02_make_tf_file.bat コピーして置換
    sed(r'C:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\tf_api_train_base_files\02_make_tf_file.bat'
        , 'PATH_TO_BE_CONFIGURED'
        , args['PATH_TO_BE_CONFIGURED']
        , out_path=args['02_make_tf_file'])
    sed(args['02_make_tf_file'], 'IMAGE_DIR', args['IMAGE_DIR'], out_path=args['02_make_tf_file'])

    # 03_sed_config_file.bat コピーして置換
    sed(r'C:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\tf_api_train_base_files\03_sed_config_file.bat'
        , 'PATH_TO_BE_CONFIGURED'
        , args['PATH_TO_BE_CONFIGURED']
        , out_path=args['03_sed_config_file'])

    # 04_1_run_train.bat コピーして置換
    sed(r'C:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\tf_api_train_base_files\04_1_run_train.bat'
        , 'PATH_TO_BE_CONFIGURED'
        , args['PATH_TO_BE_CONFIGURED']
        , out_path=args['04_1_run_train'])

    # 04_2_run_tensorboard.bat コピーして置換
    sed(r'C:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\tf_api_train_base_files\04_2_run_tensorboard.bat'
        , 'PATH_TO_BE_CONFIGURED'
        , args['PATH_TO_BE_CONFIGURED']
        , out_path=args['04_2_run_tensorboard'])

    # 05_deploy_trained_model.bat コピーして置換
    sed(r'C:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\tf_api_train_base_files\05_deploy_trained_model.bat'
        , 'PATH_TO_BE_CONFIGURED'
        , args['PATH_TO_BE_CONFIGURED']
        , out_path=args['05_deploy_trained_model'])

    # 予測用画像path1件取得
    jpg_list = glob.glob( args['PRED_IMG_DIR']+'/*jpg' )
    png_list = glob.glob( args['PRED_IMG_DIR']+'/*png' )
    img_list = [*jpg_list, *png_list]
    PRED_IMG_PATH = args['PRED_IMG_DIR']+'/'+str(pathlib.Path(img_list[0]).name)
    #print(PRED_IMG_PATH)
    # 06_predict_1img_test.bat コピーして置換
    sed(r'C:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\tf_api_train_base_files\06_predict_1img_test.bat'
        , 'PATH_TO_BE_CONFIGURED'
        , args['PATH_TO_BE_CONFIGURED']
        , out_path=args['06_predict_1img_test'])
    sed(args['06_predict_1img_test'], 'PRED_IMG_PATH', PRED_IMG_PATH, out_path=args['06_predict_1img_test'])

    # 予測用動画path1件取得
    avi_list = glob.glob( args['PRED_IMG_DIR']+'/*avi' )
    mp4_list = glob.glob( args['PRED_IMG_DIR']+'/*mp4' )
    video_list = [*avi_list, *mp4_list]
    PRED_VIDEO_PATH = args['PRED_IMG_DIR']+'/'+str(pathlib.Path(img_list[0]).name)
    # 06_predict_1video_test.bat コピーして置換
    sed(r'C:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\tf_api_train_base_files\06_predict_1video_test.bat'
        , 'PATH_TO_BE_CONFIGURED'
        , args['PATH_TO_BE_CONFIGURED']
        , out_path=args['06_predict_1video_test'])
    sed(args['06_predict_1video_test'], 'PRED_VIDEO_PATH', PRED_VIDEO_PATH, out_path=args['06_predict_1video_test'])

    # 07_predict_keras.bat コピーして置換
    sed(r'C:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\tf_api_train_base_files\07_predict_keras.bat'
        , 'PATH_TO_BE_CONFIGURED'
        , args['PATH_TO_BE_CONFIGURED']
        , out_path=args['07_predict_keras'])
    sed(args['07_predict_keras'], 'PRED_IMG_DIR', args['PRED_IMG_DIR'], out_path=args['07_predict_keras'])
    sed(args['07_predict_keras'], 'PATH_KERAS_MODEL', args['PATH_KERAS_MODEL'], out_path=args['07_predict_keras'])
    sed(args['07_predict_keras'], 'KERAS_MODEL_HEIGHT', str(args['KERAS_MODEL_HEIGHT']), out_path=args['07_predict_keras'])
    sed(args['07_predict_keras'], 'KERAS_MODEL_WIDTH', str(args['KERAS_MODEL_WIDTH']), out_path=args['07_predict_keras'])
    sed(args['07_predict_keras'], 'PATH_KERAS_DICT_CLASS_TSV', args['PATH_KERAS_DICT_CLASS_TSV'], out_path=args['07_predict_keras'])
    sed(args['07_predict_keras'], 'KERAS_CUSTOM_OBJECTS', args['KERAS_CUSTOM_OBJECTS'], out_path=args['07_predict_keras'])

if __name__ == "__main__":
    main()
