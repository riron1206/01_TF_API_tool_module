# 01_TF_API_tool_module
tensorflow object detection API��windows���łȂ邾���ȒP�ɓ��������߂̃c�[���Q for Windows(GeForce 1080)  

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
conda install -c conda-forge pandas scikit-learn jupyter
conda install -c conda-forge pandas scikit-learn jupyter Cython Protobuf Pillow lxml Matplotlib tqdm future graphviz pydot pytest pyperclip networkx selenium beautifulsoup4 cssselect openpyxl pypdf2 python-docx requests tweepy textblob seaborn scikit-image imbalanced-learn colorlog sqlalchemy papermill opencv shapely imageio git shap eli5 umap-learn plotly ipysheet bqplot rise bokeh jupyter_contrib_nbextensions yapf flask cx_oracle rdkit tifffile xlsxwriter
conda install -c conda-forge numba=0.38.1  opencv
```

## Usage
- 00_sed_cp_base_files.bat�ɏo�̓f�B���N�g����train�摜�f�B���N�g���̃p�X������00���珇�Ԃ�bat���s����Ίw�K���s�ł���
- 00_sed_cp_base_files.bat�����R�s�[���A���s�����01-��notebook��bat�̃R�s�[���J�����g�f�B���N�g���ɍ����
- �t�@�C���̓T�N���G�f�B�^�ŊJ�����ƁBbat�t�@�C���̕����R�[�h��UTF8�łȂ���00_sed_cp_base_files.bat�G���[�ɂȂ�

## Author
- Github: [riron1206](https://github.com/riron1206)