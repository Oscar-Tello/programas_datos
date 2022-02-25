import scattertext as st
import spacy
from pprint import pprint

convention_df = st.SampleCorpora.ConventionData2012.get_data()
convention_df.iloc[0]

nlp = spacy.load('en_core_web_sm')
corpus = st.CorpusFromPandas(
    convention_df,
    category_col='party',
    text_col='text',
    nlp=nlp
).build()

print(list(corpus.get_scaled_f_scores_vs_background().index[:10]))

term_freq_df = corpus.get_term_freq_df()
term_freq_df['Democratic Score'] = corpus.get_scaled_f_scores('democrat')
pprint(list(term_freq_df.sort_values(by='Democratic Score', ascending=False).index[:10]))

term_freq_df['Republican Score'] = corpus.get_scaled_f_scores('republican')
pprint(list(term_freq_df.sort_values(by='Republican Score', ascending=False).index[:10]))

html = st.produce_scattertext_explorer(
    corpus,
    category='democrat',
    category_name='Democratic',
    not_category_name='Republican',
    width_in_pixels=1000,
    metadata=convention_df['speaker']
)

open("Convention-Visualization.html", 'wb').write(html.encode('utf-8'))