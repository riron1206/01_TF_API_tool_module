@rem 作成日2019/8/12 tfAPIの基本ファイル(02_make_tf_file.batとか)のファイルパスをコピーして置換する
@rem 区切り文字は「/」でないと置換できない！！！

@rem 置換する各引数について
@rem --PATH_TO_BE_CONFIGURED	: 学習の出力先ディレクトリパス
@rem --IMAGE_DIR				: train setの画像ディレクトリ
@rem --PATH_KERAS_MODEL			: 予測で使うkerasの画像分類モデルのパス
@rem --KERAS_MODEL_HEIGHT		: 予測で使うkerasの画像分類モデルの入力層の大きさ
@rem --KERAS_MODEL_WIDTH		: 予測で使うkerasの画像分類モデルの入力層の大きさ
@rem --PATH_KERAS_DICT_CLASS_TSV: 予測で使うkerasの画像分類モデルのクラス名とクラスidのtsvファイル
@rem --KERAS_CUSTOM_OBJECTS		: 予測で使うkerasの画像分類モデルロード時に必要なcustom_objects。OctConv2Dとか。デフォルトのNoneならなしにする
@rem --PRED_IMG_DIR				: 予測する画像ディレクトリ

call activate tfgpu_v1-11
call python C:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\tf_api_tool_module\sed_base_files.py ^
--PATH_TO_BE_CONFIGURED D:/work/kaggle_kuzushiji-recognition/work/object_detection/20190812 ^
--IMAGE_DIR D:/work/kaggle_kuzushiji-recognition/OrigData/kuzushiji-recognition/train_images ^
--PATH_KERAS_MODEL D:/work/kaggle_kuzushiji-recognition/work/classes/20190816/best_val_acc.h5 ^
--KERAS_MODEL_HEIGHT 32 ^
--KERAS_MODEL_WIDTH 32 ^
--PATH_KERAS_DICT_CLASS_TSV D:/work/kaggle_kuzushiji-recognition/work/classes/20190816/tfAPI_dict_class.tsv ^
--KERAS_CUSTOM_OBJECT OctConv2D ^
--PRED_IMG_DIR D:/work/kaggle_kuzushiji-recognition/GrayImg/test_images

@rem pause