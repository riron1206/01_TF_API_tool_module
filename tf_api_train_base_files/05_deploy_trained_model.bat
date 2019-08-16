@rem 作成日2019/8/16 predict用に学習したモデルをinference_graphディレクトリにデプロイ。学習したモデルのstep番号(model.ckpt-196725)は手で修正すること

call activate tfgpu_v1-11
call python C:\\Users\\shingo\\Git\\models\\research\\object_detection\\export_inference_graph.py ^
--input_type image_tensor ^
--pipeline_config_path PATH_TO_BE_CONFIGURED/faster_rcnn_inception_resnet_v2_atrous_cosine_lr_coco_edit_sed.config ^
--trained_checkpoint_prefix PATH_TO_BE_CONFIGURED/log_train/model.ckpt-196725 ^
--output_directory PATH_TO_BE_CONFIGURED/inference_graph
