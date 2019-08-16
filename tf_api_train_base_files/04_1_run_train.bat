@rem 作成日2019/8/11 Object-Detection-APIで学習実行

call activate tfgpu_v1-11
call python C:/Users/shingo/Git/models/research/object_detection/train.py ^
--logtostderr ^
--train_dir=PATH_TO_BE_CONFIGURED/log_train ^
--pipeline_config_path=PATH_TO_BE_CONFIGURED/faster_rcnn_inception_resnet_v2_atrous_cosine_lr_coco_edit_sed.config

pause