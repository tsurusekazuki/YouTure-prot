import requests
import cv2
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd

from IPython.display import display,Image,display_jpeg

from google.colab import drive
drive.mount('/content/drive/')
path = '/content/drive/My Drive/Colab Notebooks/youtube_app'

df = pd.read_table(path+'/JPvideos.csv', delimiter=",")
df_thumbnail_link = pd.DataFrame({'video_id': df['video_id'], 'thumbnail_link': df['thumbnail_link']})
df_thumbnail_link['thumbnail_link'] = df_thumbnail_link['thumbnail_link'].str.replace('default.jpg', 'maxresdefault.jpg')
df_thumbnail_link.head(3)


# https://sonaeru-blog.com/image-ai/#i-3

class Youtube_DB:

  def __init__(self, ):
    self.db = [[], [], [], [], []]
    
  def img_download(self, video_id, thumbnail_link):
    response = requests.get(thumbnail_link, allow_redirects=False)
    if response.status_code == 200:
      img = response.content
      display_jpeg(Image(img))

      
  def add_user_db(self, user_id):
    nan = float("nan")


  def print_db(self, ):
    print([print(len(i)) for i in self.db])
    print(self.db)

def download_img(url, file_name):
  # print(url, file_name)
  r = requests.get(url, stream=True)
  if r.status_code == 200:
    with open(path+'/imagiai/images/' +file_name+'.jpg', 'wb') as f:
      f.write(r.content)
      download_list.append(file_name)

download_list = []
for i in range(20):
  download_img(df_thumbnail_link['thumbnail_link'][i], df_thumbnail_link['video_id'][i])
print(download_list)
