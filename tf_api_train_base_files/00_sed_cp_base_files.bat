@rem 作成日2019/8/12 tfAPIの基本ファイル(02_make_tf_file.batとか)のファイルパスをコピーして置換する。PATH_TO_BE_CONFIGUREDとIMAGE_DIR の区切り文字は「/」でないと置換できない

call activate tfgpu_v1-11
call python C:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\tf_api_tool_module\sed_base_files.py ^
--PATH_TO_BE_CONFIGURED=D:/work/kaggle_kuzushiji-recognition/work/object_detection/20190812 ^
--IMAGE_DIR=D:/work/kaggle_kuzushiji-recognition/OrigData/kuzushiji-recognition/train_images

pause