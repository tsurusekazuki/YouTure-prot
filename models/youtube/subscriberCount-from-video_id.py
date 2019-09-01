import requests
import json
import pandas as pd
import csv

from google.colab import drive
drive.mount('/content/drive/')
path = '/content/drive/My Drive/Colab Notebooks/youtube_app'

df = pd.read_table(path+'/JPvideos.csv', delimiter=",")
new_df = df[df['video_id'] != '#NAME?']
new_df = new_df.reset_index(drop=True)
new_df.head(3)

# チャンネルID取得
def channel_id_api(video_id):
  url = 'https://www.googleapis.com/youtube/v3/videos?id={}&key=AIzaSyA4P8WXNBtsuRcnWJDvjYNCmcBxSf-etXU&fields=items(id,snippet(channelId))&part=snippet'.format(video_id)

  r = requests.get(url).json()
  print(r)
  if r['items'] != []:
    print('ok')
    r_channel = r['items'][0]['snippet']['channelId']
    channel_id.append([video_id, r_channel])
    print(r_channel)

channel_id = []
for video in new_df['video_id']:
  channel_id_api(new_df['video_id'][i])


# 登録者数取得
def subscriber_api(channel_id):
  url = 'https://www.googleapis.com/youtube/v3/channels?part=statistics&id={}&key=AIzaSyD4tioqcE572eL5t-xIoKe5wW7Npk8BAGU&part=items&fields=items(statistics(subscriberCount))'.format(channel_id)
  r = requests.get(url).json()
  print(r)
  if r['items'] != []:
    print('ok')
    r_subscriberCount = r['items'][0]['statistics']['subscriberCount']
    subscriber_count.append([channel_id, r_subscriberCount])
    print(r_subscriberCount)

subscriber_count = []

for video in df['video_id']:
  subscriber_api(df['channelId'][i])
