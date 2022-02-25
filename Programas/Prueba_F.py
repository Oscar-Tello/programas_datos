import pandas as pd
import numpy as np
import scattertext as st
import spacy
from IPython.display import IFrame
from IPython.core.display import display, HTML
from scipy.stats import hmean
from scipy.stats import norm
display(HTML("<style>.container { width:98% !important; }</style>"))
import matplotlib.pyplot as plt

assert st.__version__ >= '0.0.2.25'

rdf = st.SampleCorpora.RottenTomatoes.get_data()
rdf['category_name'] = rdf['category'].apply(lambda x: {'plot': 'Plot', 'rotten': 'Negative', 'fresh': 'Positive'} [x])
print(rdf.category_name.value_counts())
rdf[['text', 'movie_name', 'category_name']].head()

corpus = (st.CorpusFromPandas(rdf,
                              category_col='category_name',
                              text_col='text',
                              nlp=st.whitespace_nlp_with_sentences)
          .build())
corpus.get_term_freq_df().to_csv('term_freqs2.csv')
unigram_corpus = corpus.get_unigram_corpus()