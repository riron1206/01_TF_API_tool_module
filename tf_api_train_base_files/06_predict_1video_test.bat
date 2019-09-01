@rem 作成日2019/9/1 predict_testディレクトリに動画1枚predict
@rem PATH_TO_VIDEO は評価したい動画のpathに変更すること
@rem COCOとかのデフォルトのlabel_map.pbtxt はC:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\models\research\object_detection\data にある
@rem threshold は予測スコアの閾値。0-1の間で変更すること

call activate tfgpu_v1-11
call python C:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\tf_api_tool_module\\Object_detection_video_edit.py ^
--PATH_TO_CKPT PATH_TO_BE_CONFIGURED/inference_graph/frozen_inference_graph.pb ^
--PATH_TO_LABELS PATH_TO_BE_CONFIGURED/label_map.pbtxt ^
--PATH_TO_VIDEO PRED_VIDEO_PATH ^
--NUM_CLASSES 1 ^
--output PATH_TO_BE_CONFIGURED/predict_test ^
--threshold 0.7

pause
