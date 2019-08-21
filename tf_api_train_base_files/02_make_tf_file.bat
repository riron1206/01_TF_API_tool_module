@rem 作成日2019/8/11 train.record作成する

call activate tfgpu_v1-11
call python C:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\tf_api_tool_module\generate_tfrecord.py ^
--csv_xywh_id PATH_TO_BE_CONFIGURED/xywh_train.csv ^
--csv_class_name PATH_TO_BE_CONFIGURED/classes_text.csv ^
--image_dir IMAGE_DIR ^
--output_path PATH_TO_BE_CONFIGURED/train.record
