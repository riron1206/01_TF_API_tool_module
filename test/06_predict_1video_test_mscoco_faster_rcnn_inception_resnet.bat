@rem �쐬��2019/9/1 predict_test�f�B���N�g���ɓ���1��predict
@rem PATH_TO_VIDEO �͕]�������������path�ɕύX���邱��
@rem COCO�Ƃ��̃f�t�H���g��label_map.pbtxt ��C:/Users/shingo/jupyter_notebook/tfgpu_v1-11_work/01_TF_API_tool_module/models/research/object_detection/data �ɂ���
@rem threshold �͗\���X�R�A��臒l�B0-1�̊ԂŕύX���邱��

call activate tfgpu_v1-11
call python C:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\tf_api_tool_module\\Object_detection_video_edit.py ^
--PATH_TO_CKPT D:/work/zoo_model/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb ^
--PATH_TO_LABELS C:/Users/shingo/jupyter_notebook/tfgpu_v1-11_work/01_TF_API_tool_module/models/research/object_detection/data/mscoco_label_map.pbtxt ^
--NUM_CLASSES 90 ^
--output output ^
--threshold 0.7 ^
--PATH_TO_VIDEO input/IMG_1320.MOV
@rem --PATH_TO_VIDEO input/LKHI3024.MP4 # 2019/8 ���엷�s
@rem --PATH_TO_VIDEO input/IMG_1311.MOV # 2019/8/31�{���ƃ_�[�c1

pause
