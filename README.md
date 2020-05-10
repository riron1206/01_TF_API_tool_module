# 01_TF_API_tool_module
tensorflow object detection API ( faster_rcnn_inception_resnet_v2 ) ��windows���łȂ邾���ȒP�ɓ��������߂̃c�[���Q

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
- tf_api_tool_module�͊w�K�Ŏg��python�X�N���v�g
- tf_api_train_base_files��tf_api_tool_module�ŕK�v�ȃt�@�C�����쐬����bat�t�@�C����notebook
- models��2018/05/19���_��tensorflow object detection API(https://github.com/tensorflow/models/tree/master/research)

- tf_api_train_base_files/00_sed_cp_base_files.bat�ɏo�̓f�B���N�g����train�摜�f�B���N�g���̃p�X��keras�̉摜���ރ��f���̃p�X�Ȃǂ����A00���珇�Ԃ�bat�t�@�C�������s����Ίw�K���s�ł���
	- bat�t�@�C���̓T�N���G�f�B�^�ŊJ�����ƁB�����R�[�h��UTF8�łȂ���tf_api_train_base_files/00_sed_cp_base_files.bat���G���[�ɂȂ�
	- ����bat�t�@�C�����s���ɕ�����������ꍇ��SIFT-JIS�ɕύX���邱�Ɓi��{�A�o�b�`�t�@�C���̕����R�[�h��SIFT-JIS����Ȃ��Ⴞ�߁j

## Author
- Github: [riron1206](https://github.com/riron1206)