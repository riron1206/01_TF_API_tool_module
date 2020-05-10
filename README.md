# 01_TF_API_tool_module
tensorflow object detection API ( faster_rcnn_inception_resnet_v2 ) をwindows環境でなるだけ簡単に動かすためのツール群

## Setup
- Anaconda 4.4.10: https://www.anaconda.com/distribution/
- NVIDIA Driver for GeForce 1080: http://www.nvidia.co.jp/Download/index.aspx?lang=jp
- Visual Studio 2015 Update 3(Visual Studio Community 2015 with Update 3): https://my.visualstudio.com/Downloads  
	- Custom installation of Visual Studio 2015
- Visual C ++ Redistributable Package for Visual Studio 2015: https://www.microsoft.com/ja-JP/download/details.aspx?id=48145
- CUDA Toolkit 9.0: https://developer.nvidia.com/cuda-90-download-archive  
	- Installation execution except "Visual studio integration"
- cuDNN v7.0.5: https://developer.nvidia.com/rdp/cudnn-download
```bash
conda create -n tfgpu_v1-11
activate tfgpu_v1-11
conda install -c anaconda tensorflow-gpu=1.11
conda install -c conda-forge keras=2.2.4
conda install -c conda-forge pandas scikit-learn jupyter Cython Protobuf Pillow lxml Matplotlib tqdm future graphviz pydot pytest pyperclip networkx selenium beautifulsoup4 cssselect openpyxl pypdf2 python-docx requests tweepy textblob seaborn scikit-image imbalanced-learn colorlog sqlalchemy papermill opencv shapely imageio git shap eli5 umap-learn plotly ipysheet bqplot rise bokeh jupyter_contrib_nbextensions yapf flask cx_oracle rdkit tifffile xlsxwriter
conda install -c conda-forge numba=0.38.1 opencv
```

## Usage
- tf_api_tool_moduleは学習で使うpythonスクリプト
- tf_api_train_base_filesはtf_api_tool_moduleで必要なファイルを作成するbatファイルとnotebook
- modelsは2018/05/19時点のtensorflow object detection API(https://github.com/tensorflow/models/tree/master/research)

- tf_api_train_base_files/00_sed_cp_base_files.batに出力ディレクトリとtrain画像ディレクトリのパスやkerasの画像分類モデルのパスなどを入れ、00から順番にbatファイルを実行すれば学習実行できる
	- batファイルはサクラエディタで開くこと。文字コードがUTF8でないとtf_api_train_base_files/00_sed_cp_base_files.batがエラーになる
	- 他のbatファイル実行時に文字化けする場合はSIFT-JISに変更すること（基本、バッチファイルの文字コードはSIFT-JISじゃなきゃだめ）

## Author
- Github: [riron1206](https://github.com/riron1206)