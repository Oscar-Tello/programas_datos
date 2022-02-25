import scattertext as st
import umap
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from gensim.models.word2vec import Word2Vec

convention_df = st.SampleCorpora.ConventionData2012.get_data()
convention_df['parse'] = convention_df['text'].apply(st.whitespace_nlp_with_sentences)

unigram_corpus = (st.CorpusFromParsedDocuments(convention_df,
                                               category_col='party',
                                               parsed_col='parse')
                  .build().get_stoplisted_unigram_corpus())
topic_model = st.SentencesForTopicModeling(unigram_corpus).get_topics_from_model(
    Pipeline([
        ('tfidf', TfidfTransformer(sublinear_tf=True)),
        ('nmf', (NMF(n_components=100, alpha=.1, l1_ratio=.5, random_state=0)))
    ]),
    num_terms_per_topic=20
)

topic_feature_builder = st.FeatsFromTopicModel(topic_model)

topic_corpus = st.CorpusFromParsedDocuments(
    convention_df,
    category_col='party',
    parsed_col='parse',
    feats_from_spacy_doc=topic_feature_builder
).build()

html = st.produce_scattertext_explorer(
    topic_corpus,
    category='democrat',
    category_name='Democratic',
    not_category_name='Republican',
    width_in_pixels=1000,
    metadata=convention_df['speaker'],
    use_non_text_features=True,
    use_full_doc=True,
    pmi_threshold_coefficient=0,
    topic_model_term_lists=topic_feature_builder.get_top_model_term_lists(),
    topic_model_preview_size=20
)

open('Demo_NMF_Topic_Model.html', 'wb').write(html.encode('utf-8'))

term_list = ['obama', 'romney', 'democrats', 'republicans', 'health', 'military', 'taxes',
             'education', 'olympics', 'auto', 'iraq', 'iran', 'israel', 'money']

unigram_corpus = (st.CorpusFromParsedDocuments(convention_df,
                                               category_col='party',
                                               parsed_col='parse')
                  .build().get_stoplisted_unigram_corpus())

topic_model = (st.SentencesForTopicModeling(unigram_corpus)
               .get_topics_from_terms(term_list,
                                      scorer=st.RankDifference(),
                                      num_terms_per_topic=20))

topic_feature_builder = st.FeatsFromTopicModel(topic_model)

topic_corpus = st.CorpusFromParsedDocuments(
    convention_df,
    category_col='party',
    parsed_col='parse',
    feats_from_spacy_doc=topic_feature_builder
).build()

html = st.produce_scattertext_explorer(
    topic_corpus,
    category='democrat',
    category_name='Democratic',
    not_category_name='Republican',
    width_in_pixels=1000,
    metadata=convention_df['speaker'],
    use_non_text_features=True,
    use_full_doc=True,
    pmi_threshold_coefficient=0,
    topic_model_term_lists=topic_feature_builder.get_top_model_term_lists(),
    topic_model_preview_size=20
)

open('Demo_Word_List_Topic_Model.html', 'wb').write(html.encode('utf-8'))

# Creating T-SNE-style word embedding projection plots

convention_df = st.SampleCorpora.ConventionData2012.get_data()
convention_df['parse'] = convention_df['text'].apply(st.whitespace_nlp_with_sentences)

corpus = (st.CorpusFromParsedDocuments(convention_df, category_col='party', parsed_col='parse')
          .build().get_stoplisted_unigram_corpus())



# Using SVD to visualize any kind of word embeddings

corpus = (st.CorpusFromParsedDocuments(convention_df,
                                       category_col='party',
                                       parsed_col='parse')
          .build()
          .get_stoplisted_unigram_corpus()
          .remove_infrequent_words(minimum_term_count=3, term_ranker=st.OncePerDocFrequencyRanker))

from sklearn.feature_extraction.text import TfidfTransformer

embeddings = TfidfTransformer().fit_transform(corpus.get_term_doc_mat())
embeddings.shape
corpus.get_num_docs(), corpus.get_num_terms()
embeddings = embeddings.T
embeddings.shape

from scipy.sparse.linalg import  svds

U, S, VT = svds(embeddings,k = 3, maxiter=20000, which='LM')
U.shape
S.shape
VT.shape

x_dim = 0; y_dim = 1;
projection = pd.DataFrame({'term':corpus.get_terms(),
                           'x':U.T[x_dim],
                           'y':U.T[y_dim]}).set_index('term')

html = st.produce_pca_explorer(corpus,
                               category='democrat',
                               category_name='Democratic',
                               not_category_name='Republican',
                               projection=projection,
                               metadata=convention_df['speaker'],
                               width_in_pixels=1000,
                               x_dim=x_dim,
                               y_dim=y_dim)

open('Demo_Embeddings_SVD_0_1.html', 'wb').write(html.encode('utf-8'))


html = st.produce_pca_explorer(corpus,
                               category='democrat',
                               category_name='Democratic',
                               not_category_name='Republican',
                               projection=projection,
                               metadata=convention_df['speaker'],
                               width_in_pixels=1000,
                               scaler=st.scale_neg_1_to_1_with_zero_mean,
                               x_dim=x_dim,
                               y_dim=y_dim)

open('Demo_Embeddings_SVD_0_1_Scale_Neg_1_To_1_With_Zero_Mean.html', 'wb').write(html.encode('utf-8'))