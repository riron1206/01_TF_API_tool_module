@rem 作成日2019/8/16 指定ディレクトリ全画像predict。kerasのモデルで分類もする
@rem PATH_TO_IMAGE は評価したい画像ディレクトリのpathに変更すること
@rem --is_img_save ^のオプションつけたら検出画像も作成するが、1件処理するのに20-30秒かかる。無ければ1件5-9秒で処理できる
@rem threshold は領域予測スコアの閾値。0-1の間で変更すること
@rem class_threshold はkerasの分類モデルのスコアの閾値。0-1の間で変更すること

call activate tfgpu_v1-11
call python C:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\tf_api_tool_module\\Object_detection_image_edit_module.py ^
--PATH_TO_CKPT PATH_TO_BE_CONFIGURED/inference_graph/frozen_inference_graph.pb ^
--PATH_TO_LABELS PATH_TO_BE_CONFIGURED/label_map.pbtxt ^
--PATH_TO_IMAGE D:\work\kaggle_kuzushiji-recognition\OrigData\kuzushiji-recognition\test_images\test_0adbe8a5.jpg ^
--NUM_CLASSES 1 ^
--output PATH_TO_BE_CONFIGURED/predict_keras_all ^
--is_overwrite True ^
--tfapi_threshold 0.6 ^
--class_model_path PATH_KERAS_MODEL ^
--class_threshold 0.0001 ^
--model_height KERAS_MODEL_HEIGHT ^
--model_width KERAS_MODEL_WIDTH ^
--dict_class_tsv PATH_KERAS_DICT_CLASS_TSV ^
--custom_objects KERAS_CUSTOM_OBJECTS

@rem pause
