from lightning.classification import CDClassifier
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import scattertext as st

newsgroups_train = fetch_20newsgroups(
	subset='train',
	remove=('headers', 'footers', 'quotes')
)

