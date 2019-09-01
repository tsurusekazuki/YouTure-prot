import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from janome.tokenizer import Tokenizer
import random

# data = pd.read_csv('../data/view_title_100.csv', names = ('number', '100_view_title'))
# data = data.dropna(how='any')
# data = data.astype({'number': int})
# data = pd.DataFrame({'100_view_title': data['100_view_title']})
data = pd.read_csv('../data/r_100_view_title.csv')
data = data.drop('Unnamed: 0', axis=1)
data = data.sample(n=100)
text = input()

corpus = []
cr = pd.Series([text], index=data.columns)


data = data.append(cr, ignore_index=True)

def title_split(title):
    t_wakati = Tokenizer(wakati=True)
    text = t_wakati.tokenize(title)
    text = ' '.join(text)
    return text

def sentence_to_vect(ts):
    vectorizer = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    transformer = TfidfTransformer()

    tf = vectorizer.fit_transform(ts)
    tfidf = transformer.fit_transform(tf)
    return tfidf.toarray()

split_corpus = []

corpus = data['100_view_title']
for title in corpus:
    split_corpus.append(title_split(title))


tfidf_array = sentence_to_vect(split_corpus)
cs = cosine_similarity(tfidf_array,tfidf_array)  # cos類似度計算
cs = cs[-1]
cs = list(map(lambda x: x * 100, cs))
data['percentage'] = cs
data = data.drop(100, axis=0)
data = data.sort_values('percentage', ascending=False)

print(data[:10])
print('-------------------------------------')
print(data.iloc[0, 0])



