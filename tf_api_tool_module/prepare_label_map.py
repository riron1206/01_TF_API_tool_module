"""
tfAPIが読むためのクラスidとクラス名の対応ファイル label_map.pbtxt 作成するためのコード
"""
import os, sys

def make_label_map_pbtxt(classes_text_df, label_map_path):
    """
    クラスidとクラス名のデータフレームからlabel_map.pbtxtの作成
    Args:
        classes_text_df: class_id，class_name のデータフレーム
        label_map_path: 出力するlabel_map.pbtxt のパス
    """
    if os.path.isfile(label_map_path):
        os.remove(label_map_path)

    with open(label_map_path, mode='a') as f:
        for id, name in zip(classes_text_df['id'], classes_text_df['name']):
            f.write("item {\n")
            f.write("  id: "+str(id)+"\n")
            f.write("  name: '"+str(name)+"'\n")
            f.write("}\n")
