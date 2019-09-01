!pip3 install imageai --upgrade
#!wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5
#"!wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo-tiny.h5
!wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/DenseNet-BC-121-32.h5
#!wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_weights_tf_dim_ordering_tf_kernels.h5

import numpy as np
import pandas as pd
import csv

from IPython.display import display,Image,display_jpeg

from google.colab import drive
drive.mount('/content/drive/')
path = '/content/drive/My Drive/Colab Notebooks/youtube_app'

# ダウンロードリスト取得
f = open(path+'/imagiai/images/'+'downloadlist.csv', "r")
reader = csv.reader(f)
download_data = [ e for e in reader ]
print(download_data)
f.close()

# video_id 取得
df = pd.read_table(path+'/JPvideos.csv', delimiter=",")
df_thumbnail_link = pd.DataFrame({'video_id': df['video_id'], 'thumbnail_link': df['thumbnail_link']})
df_thumbnail_link['thumbnail_link'] = df_thumbnail_link['thumbnail_link'].str.replace('default.jpg', 'maxresdefault.jpg')
df_thumbnail_link.head(3)

# 画像認識
from imageai.Prediction import ImagePrediction

prediction = ImagePrediction()
prediction.setModelTypeAsDenseNet()
prediction.setModelPath(os.path.join(execution_path, "DenseNet-BC-121-32.h5"))
prediction.loadModel()
def predict_img(video_id):
  predictions, probabilities = prediction.predictImage(path+'/imagiai/images/' + video_id +'.jpg', result_count=3)
  predict_list.append([[video_id], [predictions]])


# 処理
download_list = []
for video_name in download_data[0]:
  download_list.append(video_name)

predict_list = []
for download in download_list:
  predict_img(download)

