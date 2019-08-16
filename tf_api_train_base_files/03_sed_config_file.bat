@rem 作成日2019/8/11 configファイルのファイルパス置換してコピー作成する
@rem val_record_path、eval_num_examples 設定しているが eval.py を実行して評価しないとvalidation set確認できない。つまり、Kerasのようにtrainだけでvalidation setの評価はできない
@rem first_stage_max_number_of_boxes が領域検出の最大box数。1画像に最大でboxが何個あり得るか検討して設定すること。max_detections_per_class、max_total_detections は分類も含めての検出数なので、first_stage_max_number_of_boxesに合わせること

call activate tfgpu_v1-11
call python C:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\tf_api_tool_module\prepare_coco_config.py ^
--config_path C:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\tf_api_train_base_files\faster_rcnn_inception_resnet_v2_atrous_cosine_lr_coco_edit.config ^
--out_path PATH_TO_BE_CONFIGURED/faster_rcnn_inception_resnet_v2_atrous_cosine_lr_coco_edit_sed.config ^
--train_record_path PATH_TO_BE_CONFIGURED/train.record ^
--val_record_path PATH_TO_BE_CONFIGURED/validation.record ^
--label_map_pbtxt_path PATH_TO_BE_CONFIGURED/label_map.pbtxt ^
--eval_num_examples 0 ^
--num_classes 1 ^
--max_detections_per_class 650 ^
--max_total_detections 650 ^
--first_stage_max_number_of_boxes 650

