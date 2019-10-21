# YouTure

## Description

YouTuberにとって動画の再生数を増やすというのは重要なことである。よってあらかじめ動画の再生数、高評価数がどれくらいになるのか？というのを
知ることができれば、効率よく再生数、高評価を高く取得できる動画の作成、改善に繋がるためこのアプリを開発した。

[YouTure](https://youturepjt.herokuapp.com/)

## Demo

![demo](https://user-images.githubusercontent.com/38784824/64070418-16240500-cc9a-11e9-8915-517120c80889.gif)

## Feature

- 動画の予測再生回数、予測高評価数を表示する。
- 入力したタイトルと似ているタイトルを表示する。
- 入力したサムネイルにある対象をキーワードとして渡すと、似ているサムネイルの動画を表示する。

## Usage
 
1. 自分の作成した動画のタイトルをタイトル入力フォームに入力する。
2. 自分の作成した動画のカテゴリをカテゴリ選択メニューから選択する。
3. 自分のYouTubeのチャンネル登録者数を入力する。
4. 自分の作成した動画のサムネイルにある対象となるキーワードを入力する。
5. Submitボタンを押す。

## Installation 
 
`$ git clone https://github.com/tsurusekazuki/YouTure.git`

## Get Start

```
$ cd YouTure
$ cd cgi-app
$ python -m http.server --cgi
```

## Use Model Algorithm
- DenseNet
- tf-idf
- Linear regression
- cos similarity

## Use Dataset

kaggleが公開しているデータセット: [Trending YouTube Video Statistics](https://www.kaggle.com/datasnaek/youtube-new)

## SlideShare

YouTureの発表資料: [SlideShare: YouTure](https://www.slideshare.net/tsurusekazuki/youture)
