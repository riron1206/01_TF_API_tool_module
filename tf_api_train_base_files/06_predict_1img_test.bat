@rem 作成日2019/8/16 predict_testディレクトリに画像1枚predict
@rem PATH_TO_IMAGE は評価したい画像のpathに変更すること
@rem threshold は予測スコアの閾値。0-1の間で変更すること

call activate tfgpu_v1-11
call python C:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\tf_api_tool_module\\Object_detection_image_edit.py ^
--PATH_TO_CKPT PATH_TO_BE_CONFIGURED/inference_graph/frozen_inference_graph.pb ^
--PATH_TO_LABELS PATH_TO_BE_CONFIGURED/label_map.pbtxt ^
--PATH_TO_IMAGE PRED_IMG_PATH ^
--NUM_CLASSES 1 ^
--output PATH_TO_BE_CONFIGURED/predict_test ^
--threshold 0.0

pause