@rem 作成日2019/9/1 predict_testディレクトリに動画1枚predict
@rem PATH_TO_VIDEO は評価したい動画のpathに変更すること
@rem COCOとかのデフォルトのlabel_map.pbtxt はC:/Users/shingo/jupyter_notebook/tfgpu_v1-11_work/01_TF_API_tool_module/models/research/object_detection/data にある
@rem threshold は予測スコアの閾値。0-1の間で変更すること

call activate tfgpu_v1-11
call python C:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\tf_api_tool_module\\Object_detection_video_edit.py ^
--PATH_TO_CKPT D:/work/zoo_model/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb ^
--PATH_TO_LABELS C:/Users/shingo/jupyter_notebook/tfgpu_v1-11_work/01_TF_API_tool_module/models/research/object_detection/data/mscoco_label_map.pbtxt ^
--NUM_CLASSES 90 ^
--output output ^
--threshold 0.7 ^
--PATH_TO_VIDEO input/IMG_1320.MOV
@rem --PATH_TO_VIDEO input/LKHI3024.MP4 # 2019/8 旭川旅行
@rem --PATH_TO_VIDEO input/IMG_1311.MOV # 2019/8/31宮下家ダーツ1

pause
