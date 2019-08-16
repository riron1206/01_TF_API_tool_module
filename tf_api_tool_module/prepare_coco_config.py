"""
tfAPIの学習configファイル編集するためのコード
cocoのconfigについて
"""
import os, sys
import argparse

def sed(file_path, before_str, after_str, out_path=None):
    """
    python でsed コマンド再現。指定ファイルの指定文字を置換して別名ファイル出力。
    https://qiita.com/tsukapah/items/f2235a38592a7082662b
    https://pythonmemo.hatenablog.jp/entry/2018/05/05/135955
    Arges:
        file_path: 置換するファイルパス
        before_str: 置換前の文字
        after_str: 置換後の文字
        out_path: 出力ファイルパス。Noneなら_sed をつけたファイル名で出力
    """
    import sys
    import re
    # ファイルロード
    with open(file_path, mode='r', encoding="utf-8") as f:
        body = f.read()
    # re.sub()で置換 flags=re.DOTALLを指定したことで、正規表現の.に改行コード\nも含まれるようになり、複数行が置換対象
    sed_body = re.sub(before_str, after_str, body, flags=re.DOTALL)
    # 出力ファイル名
    if out_path is None:
        out_path = re.sub('\.config', '_sed.config', file_path)
    # ファイル出力
    with open(out_path, mode='w', encoding="utf-8") as f:
        f.write(sed_body)
    #print('out_path:', out_path)
    return

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config_path", type=str, default=r"C:\Users\shingo\jupyter_notebook\tfgpu_v1-11_work\01_TF_API_tool_module\tf_api_tool_module\faster_rcnn_inception_resnet_v2_atrous_cosine_lr_coco_edit.config",
        help="path to input config file")
    ap.add_argument("-o", "--out_path", type=str, default="PATH_TO_BE_CONFIGURED/faster_rcnn_inception_resnet_v2_atrous_cosine_lr_coco_edit_sed.config",
        help="path to output config file")
    ap.add_argument("-ckpt", "--ckpt_path", type=str, default="D:/work/zoo_model/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/model.ckpt",
        help="path to model.ckpt")
    ap.add_argument("-train", "--train_record_path", type=str, default="PATH_TO_BE_CONFIGURED/train.record",
        help="path to train.record")
    ap.add_argument("-val", "--val_record_path", type=str, default="PATH_TO_BE_CONFIGURED/validation.record",
        help="path to val.record")
    ap.add_argument("-label", "--label_map_pbtxt_path", type=str, default="PATH_TO_BE_CONFIGURED/label_map.pbtxt",
        help="path to label_map.pbtxt")
    ap.add_argument("-e", "--eval_num_examples", type=int, default=0,
        help="image eval_num_examples")
    ap.add_argument("-n", "--num_classes", type=int, default=1,
        help="number of classes")
    ap.add_argument("-mdpc", "--max_detections_per_class", type=int, default=100,
        help="number of max_detections_per_class")
    ap.add_argument("-mtd", "--max_total_detections", type=int, default=100,
        help="number of max_total_detections")
    ap.add_argument("-mnb", "--max_number_of_boxes", type=int, default=300,
        help="number of max_number_of_boxes for SSD")
    ap.add_argument("-fmnb", "--first_stage_max_number_of_boxes", type=int, default=300,
        help="first_stage number of max_number_of_boxes for SSD")
    args = vars(ap.parse_args())
    config_path = args["config_path"]
    out_path = args["out_path"]
    ckpt_path = args["ckpt_path"]
    train_record_path = args["train_record_path"]
    val_record_path = args["val_record_path"]
    label_map_pbtxt_path = args["label_map_pbtxt_path"]
    eval_num_examples = str(args["eval_num_examples"])
    num_classes = str(args["num_classes"])
    max_detections_per_class = str(args["max_detections_per_class"])
    max_total_detections = str(args["max_total_detections"])
    max_number_of_boxes = str(args["max_number_of_boxes"])
    first_stage_max_number_of_boxes = str(args["first_stage_max_number_of_boxes"])

    # num_classes: 90 置換
    sed(config_path
        , 'num_classes: [0-9]+'
        , 'num_classes: '+num_classes
        , out_path=out_path)
    # fine_tune_checkpoint PATH_TO_BE_CONFIGURED/model.ckpt 置換
    sed(out_path
        , r'PATH_TO_BE_CONFIGURED/model.ckpt'
        , ckpt_path
        , out_path=out_path)
    # input_path PATH_TO_BE_CONFIGURED/mscoco_train.record 置換
    sed(out_path
        , r'PATH_TO_BE_CONFIGURED/mscoco_train.record'
        , train_record_path
        , out_path=out_path)
    # label_map_path PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt 置換
    sed(out_path
        , r'PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt'
        , label_map_pbtxt_path
        , out_path=out_path)
    # num_examples: 50 置換
    sed(out_path
        , 'num_examples: [0-9]+'
        , 'num_examples: '+eval_num_examples
        , out_path=out_path)
    # PATH_TO_BE_CONFIGURED/mscoco_val.record 置換
    sed(out_path
        , r'PATH_TO_BE_CONFIGURED/mscoco_val.record'
        , val_record_path
        , out_path=out_path)
    # max_detections_per_class (1クラスについて予測する最大box数) 置換
    sed(out_path
        , r'max_detections_per_class: [0-9]+'
        , 'max_detections_per_class: '+max_detections_per_class
        , out_path=out_path)
    # max_total_detections (1枚の画像で予測する最大box数) 置換
    sed(out_path
        , r'max_total_detections: [0-9]+'
        , 'max_total_detections: '+max_total_detections
        , out_path=out_path)
    # max_number_of_boxes (SSDで1枚の画像で予測する最大box数) 置換
    sed(out_path
        , r'max_number_of_boxes: [0-9]+'
        , 'max_number_of_boxes: '+max_number_of_boxes
        , out_path=out_path)
    # first_stage_max_proposals (first stageのオブジェクト検出領域の最大box数) 置換
    sed(out_path
        , r'first_stage_max_proposals: [0-9]+'
        , 'first_stage_max_proposals: '+first_stage_max_number_of_boxes
        , out_path=out_path)

    print('out_path:', out_path)

if __name__ == "__main__":
    main()
