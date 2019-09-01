import pandas as pd

from IPython.display import display

data = pd.read_csv('sake_dataset.csv')
display(data.head(5))

# 欠損地確認
data.isnull().sum()

# 欠損値を含む行
display(data[data.isnull().any(axis=1)])
