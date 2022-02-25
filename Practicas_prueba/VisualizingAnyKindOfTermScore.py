import spacy
import numpy as np
from gensim.models import word2vec
from scattertext import SampleCorpora, word_similarity_explorer_gensim, Word2VecFromParsedCorpus, sparse_explorer, produce_scattertext_explorer
from scattertext.CorpusFromParsedDocuments import CorpusFromParsedDocuments
from sklearn.linear_model import Lasso, LogisticRegression


nlp = spacy.load('en_core_web_sm')
convention_df = SampleCorpora.ConventionData2012.get_data()
convention_df['parsed'] = convention_df.text.apply(nlp)
corpus = CorpusFromParsedDocuments(convention_df, category_col='party', parsed_col='parsed').build()
model = word2vec.Word2Vec(size=300,
                          alpha=0.025,
                          window=5,
                          min_count=5,
                          max_vocab_size=None,
                          sample=0,
                          seed=1,
                          workers=1,
                          min_alpha=0.0001,
                          sg=1,
                          hs=1,
                          negative=0,
                          cbow_mean=0,
                          iter=1,
                          null_word=0,
                          trim_rule=None,
                          sorted_vocab=1)
html = word_similarity_explorer_gensim(corpus,
                                       category='democrat',
                                       category_name='Democratic',
                                       not_category_name='Republican',
                                       target_term='jobs',
                                       minimum_term_frequency=5,
                                       pmi_threshold_coefficient=4,
                                       width_in_pixels=1000,
                                       metadata=convention_df['speaker'],
                                       word2vec=Word2VecFromParsedCorpus(corpus, model).train(),
                                       max_p_val=0.05,
                                       save_svg_button=True)
open('./demo_gensim_similarity.html', 'wb').write(html.encode('utf-8'))


html = sparse_explorer(
    corpus,
    category='democrat',
    category_name='Democratic',
    not_category_name='Republican',
    scores=corpus.get_regression_coefs('democrat', Lasso(max_iter=10000)),
    minimum_term_frequency=5,
    pmi_threshold_coefficient=4,
    width_in_pixels=1000,
    metadata=convention_df['speaker']
)
open('Convention-Visualization-Sparse.html', 'wb').write(html.encode('utf-8'))

# Custom term positions
term_freq_df = corpus.get_term_freq_df()

def scale(ar):
    return (ar - ar.min()) / (ar.max() - ar.min())

def zero_centered_scale(ar):
    ar[ar > 0] = scale(ar[ar > 0])
    ar[ar < 0] = -scale(-ar[ar < 0])
    return (ar + 1) / 2

frequencies_scaled = scale(np.log(term_freq_df.sum(axis=1).values))
scores = corpus.get_logreg_coefs('democrat',
                                 LogisticRegression(penalty='l2', C=10, max_iter=10000, n_jobs=-1))
scores_scaled = zero_centered_scale(scores)

html = produce_scattertext_explorer(corpus,
                                    category='democrat',
                                    category_name='Democratic',
                                    not_category_name='Republican',
                                    minimum_term_frequency=5,
                                    pmi_threshold_coefficient=4,
                                    width_in_pixels=1000,
                                    x_coords=frequencies_scaled,
                                    y_coords=scores_scaled,
                                    scores=scores,
                                    sort_by_dist=False,
                                    metadata=convention_df['speaker'],
                                    x_label='Log frequency',
                                    y_label='L2-penalized logistic regression coef'
                                    )

open('Demo_Custom_Coordinates.html', 'wb').write(html.encode('utf-8'))