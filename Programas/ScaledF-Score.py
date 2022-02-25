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
corpus.get_term_freq_df().to_csv('term_freqs.csv')
unigram_corpus = corpus.get_unigram_corpus()

html = st.produce_scattertext_explorer(
    corpus,
    category='Positive',
    not_categories=['Negative'],
    sort_by_dist=False,
    metadata=rdf['movie_name'],
    term_scorer=st.RankDifference(),
    transform=st.Scalers.percentile_dense
)

file_name = 'rotten_fresh_stdense.html'
open(file_name, 'wb').write(html.encode('utf-8'))
IFrame(src=file_name, width=1300, height=700)

html = st.produce_scattertext_explorer(
    corpus,
    category='Positive',
    not_categories=['Negative'],
    sort_by_dist=False,
    metadata=rdf['movie_name'],
    term_scorer=st.RankDifference(),
)

file_name = 'rotten_fresh_st.html'
open(file_name, 'wb').write(html.encode('utf-8'))
IFrame(src=file_name, width=1300, height=700)

term_freq_df = corpus.get_unigram_corpus().get_term_freq_df()[['Positive freq', 'Negative freq']]
term_freq_df = term_freq_df[term_freq_df.sum(axis=1) > 0]

term_freq_df['pos_precision'] = (term_freq_df['Positive freq'] * 1./
                                 (term_freq_df['Positive freq'] + term_freq_df['Negative freq']))

term_freq_df['pos_freq_pct'] = (term_freq_df['Positive freq'] * 1.
                                /term_freq_df['Positive freq'].sum())

term_freq_df['pos_hmean'] = (term_freq_df
                             .apply(lambda x: (hmean([x['pos_precision'], x['pos_freq_pct']])
                                               if x['pos_precision'] > 0 and x['pos_freq_pct'] > 0
                                               else 0), axis=1))
term_freq_df.sort_values(by='pos_hmean', ascending=False).iloc[:10]

term_freq_df.pos_freq_pct.describe()

term_freq_df.pos_precision.describe()

freq = term_freq_df.pos_freq_pct.values
prec = term_freq_df.pos_precision.values
html = st.produce_scattertext_explorer(
    corpus.remove_terms(set(corpus.get_terms()) - set(term_freq_df.index)),
    category='Positive',
    not_category_name='Negative',
    not_categories=['Negative'],

    x_label='Portion of words used in positive reviews',
    original_x=freq,
    x_coords=(freq - freq.min()) / freq.max(),
    x_axis_values=[int(freq.min() * 1000) / 1000.,
                   int(freq.max() * 1000) / 1000.],

    y_label='Portion of documents containing word that are positive',
    original_y=prec,
    y_coords=(prec - prec.min()) / prec.max(),
    y_axis_values=[int(prec.min() * 1000) / 1000.,
                   int((prec.max() / 2.) * 1000) / 1000.,
                   int(prec.max() * 1000) / 1000.],
    scores=term_freq_df.pos_hmean.values,

    sort_by_dist=False,
    show_characteristic=False
)
file_name = 'not_normed_freq_prec.html'
open(file_name, 'wb').write(html.encode('utf-8'))
IFrame(src=file_name, width=1300, height=700)

def normcdf(x):
    return norm.cdf(x, x.mean(), x.std ())

term_freq_df['pos_precision_normcdf'] = normcdf(term_freq_df.pos_precision)

term_freq_df['pos_freq_pct_normcdf'] = normcdf(term_freq_df.pos_freq_pct.values)

term_freq_df['pos_scaled_f_score'] = hmean([term_freq_df['pos_precision_normcdf'], term_freq_df['pos_freq_pct_normcdf']])

term_freq_df.sort_values(by='pos_scaled_f_score', ascending=False).iloc[:10]

term_freq_df.sort_values(by='pos_scaled_f_score', ascending=True).iloc[:10]

freq = term_freq_df.pos_freq_pct_normcdf.values
prec = term_freq_df.pos_precision_normcdf.values
html = st.produce_scattertext_explorer(
    corpus.remove_terms(set(corpus.get_terms()) - set(term_freq_df.index)),
    category='Positive',
    not_category_name='Negative',
    not_categories=['Negative'],

    x_label='Portion of words used in positive reviews (norm-cdf)',
    original_x=freq,
    x_coords=(freq - freq.min()) / freq.max(),
    x_axis_values=[int(freq.min() * 1000) / 1000.,
                   int(freq.max() * 1000) / 1000.],

    y_label='documents containing word that are positive (norm-cdf)',
    original_y=prec,
    y_coords=(prec - prec.min()) / prec.max(),
    y_axis_values=[int(prec.min() * 1000) / 1000.,
                   int((prec.max() / 2.) * 1000) / 1000.,
                   int(prec.max() * 1000) / 1000.],
    scores=term_freq_df.pos_scaled_f_score.values,

    sort_by_dist=False,
    show_characteristic=False
)
file_name = 'normed_freq_prec.html'
open(file_name, 'wb').write(html.encode('utf-8'))
IFrame(src=file_name, width=1300, height=700)

# Demostracion a Second problem: low scores are low-frequency brittle terms

term_freq_df['neg_precision_normcdf'] = normcdf((term_freq_df['Negative freq'] * 1./
                                 (term_freq_df['Negative freq'] + term_freq_df['Positive freq'])))

term_freq_df['neg_freq_pct_normcdf'] = normcdf((term_freq_df['Negative freq'] * 1.
                                /term_freq_df['Negative freq'].sum()))

term_freq_df['neg_scaled_f_score'] = hmean([term_freq_df['neg_precision_normcdf'],  term_freq_df['neg_freq_pct_normcdf']])

term_freq_df['scaled_f_score'] = 0
term_freq_df.loc[term_freq_df['pos_scaled_f_score'] > term_freq_df['neg_scaled_f_score'],
                 'scaled_f_score'] = term_freq_df['pos_scaled_f_score']
term_freq_df.loc[term_freq_df['pos_scaled_f_score'] < term_freq_df['neg_scaled_f_score'],
                 'scaled_f_score'] = 1-term_freq_df['neg_scaled_f_score']
term_freq_df['scaled_f_score'] = 2 * (term_freq_df['scaled_f_score'] - 0.5)
term_freq_df.sort_values(by='scaled_f_score', ascending=False).iloc[:10]
term_freq_df.sort_values(by='scaled_f_score', ascending=True).iloc[:10]

is_pos = term_freq_df.pos_scaled_f_score > term_freq_df.neg_scaled_f_score
freq = term_freq_df.pos_freq_pct_normcdf * is_pos - term_freq_df.neg_freq_pct_normcdf * ~is_pos
prec = term_freq_df.pos_precision_normcdf * is_pos - term_freq_df.neg_precision_normcdf * ~is_pos


def scale(ar):
    return (ar - ar.min()) / (ar.max() - ar.min())


def close_gap(ar):
    ar[ar > 0] -= ar[ar > 0].min()
    ar[ar < 0] -= ar[ar < 0].max()
    return ar


html = st.produce_scattertext_explorer(
    corpus.remove_terms(set(corpus.get_terms()) - set(term_freq_df.index)),
    category='Positive',
    not_category_name='Negative',
    not_categories=['Negative'],

    x_label='Frequency',
    original_x=freq,
    x_coords=scale(close_gap(freq)),
    x_axis_labels=['Frequent in Neg',
                   'Not Frequent',
                   'Frequent in Pos'],

    y_label='Precision',
    original_y=prec,
    y_coords=scale(close_gap(prec)),
    y_axis_labels=['Neg Precise',
                   'Imprecise',
                   'Pos Precise'],

    scores=(term_freq_df.scaled_f_score.values + 1) / 2,
    sort_by_dist=False,
    show_characteristic=False
)
file_name = 'sfs_explain.html'
open(file_name, 'wb').write(html.encode('utf-8'))
IFrame(src=file_name, width=1300, height=700)

html = st.produce_frequency_explorer(
    corpus.remove_terms(set(corpus.get_terms()) - set(term_freq_df.index)),
    category='Positive',
    not_category_name='Negative',
    not_categories=['Negative'],
    term_scorer=st.ScaledFScorePresets(beta=1, one_to_neg_one=True),
    metadata = rdf['movie_name'],
    grey_threshold=0
)
file_name = 'freq_sfs.html'
open(file_name, 'wb').write(html.encode('utf-8'))
IFrame(src=file_name, width = 1300, height=700)