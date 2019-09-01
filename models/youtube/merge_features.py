import pandas as pd
import csv

from google.colab import drive
drive.mount('/content/drive/')
path = '/content/drive/My Drive/Colab Notebooks/youtube_app'



# 登録者数/カテゴリーID
df = pd.read_table(path+'/JPvideos.csv', delimiter=",")
df = pd.DataFrame({'video_id': df['video_id'], 'category_id': df['category_id'], 'views': df['views']})

subscriberCount_df = pd.read_table(path+'/all_output_subscriberCount.csv', delimiter=",")
subscriberCount_df = pd.DataFrame({'video_id': subscriberCount_df['video_id'], 'subscriberCount': subscriberCount_df['subscriberCount']})

features_df = pd.merge(df, subscriberCount_df, how="outer", left_on='video_id', right_on='video_id')
features_df = features_df.drop_duplicates('video_id')

features_df = pd.get_dummies(features_df, columns=['category_id'])

features_df = features_df.reset_index(drop='True')

features_df.to_csv('features.csv')
features_df.head()

# 物体認識
detected_tags = pd.read_table(path + "/vector_detected_tags.csv", delimiter=",")
features_detected_df = pd.merge(features_df, detected_tags, how="outer", left_on='video_id', right_on='video_id')
features_detected_df = features_detected_df.drop_duplicates('video_id')
features_detected_df = features_detected_df.reset_index(drop='True')
features_detected_df.to_csv('features_detected.csv')
