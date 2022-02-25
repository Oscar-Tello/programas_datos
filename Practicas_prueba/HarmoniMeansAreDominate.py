import scattertext as st
from scipy.stats import norm


rdf = st.SampleCorpora.RottenTomatoes.get_data()
rdf['category_name'] = rdf['category'].apply(lambda x: {'plot': 'Plot', 'rotten': 'Negative', 'fresh': 'Positive'} [x])

corpus = (st.CorpusFromPandas(rdf,
                              category_col='category_name',
                              text_col='text',
                              nlp=st.whitespace_nlp_with_sentences)
          .build())
corpus.get_term_freq_df().to_csv('term_freqs.csv')

term_freq_df = corpus.get_unigram_corpus().get_term_freq_df()[['Positive freq', 'Negative freq']]
term_freq_df = term_freq_df[term_freq_df.sum(axis=1) > 0]

def normcdf(x):
    return norm.cdf(x, x.mean(), x.std())

term_freq_df['pos_precision_normcdf'] = normcdf(term_freq_df.pos_precision)
