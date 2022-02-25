import scattertext as st
from scipy.stats import hmean


convention_df = st.SampleCorpora.ConventionData2012.get_data()
corpus = (st.CorpusFromPandas(convention_df,
                              category_col='party',
                              text_col='text',
                              nlp= st.whitespace_nlp_with_sentences)
          .build()
          .get_unigram_corpus())
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

freq = term_freq_df.pos_freq_pct.values
prec = term_freq_df.pos_precision.values
html = st.produce_scattertext_explorer(
    corpus.remove_terms(set(corpus.get_terms()) - set(term_freq_df.index)),
    category='Positive',
    not_category_name='Negative',
    not_categories=['Negative'],

    x_label='Portion of words used in positive reviews',
    original_x=freq,
    x_coords=(freq - freq.min())/ freq.max(),
    x_axis_values=[int(freq.min()*1000)/1000.,
                   int(freq.max() * 1000)/1000.],

    y_label='Portion of documents containing word that are positive',
    original_y=prec,
    y_coords=(prec - prec.min())/prec.max(),
    y_axis_values=[int(prec.min() * 1000)/1000.,
                   int((prec.max()/2.)*1000)/1000.,
                   int(prec.max() * 1000)/1000.],
    scores=term_freq_df.pos_hmean.values,

    sort_by_dist=False,
    show_characteristic=False
)

file_name = 'not_normed_freq_prec.html'
open(file_name, 'wb').write(html.encode('utf-8'))
# IFrame(src=file_name, width = 1300, height=700)
