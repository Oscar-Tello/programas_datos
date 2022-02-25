import scattertext as st
import spacy
import numpy as np

nlp = spacy.load('en_core_web_sm')

convention_df = st.SampleCorpora.ConventionData2012.get_data()

feat_builder = st.FeatsFromOnlyEmpath()
empath_corpus = st.CorpusFromParsedDocuments(convention_df,
                                             category_col='party',
                                             feats_from_spacy_doc=feat_builder,
                                             parsed_col='text').build()

html = st.produce_scattertext_explorer(empath_corpus,
                                       category='democrat',
                                       category_name='Democratic',
                                       not_category_name='Republican',
                                       width_in_pixels=1000,
                                       metadata=convention_df['speaker'],
                                       use_non_text_features=True,
                                       use_full_doc=True,
                                       topic_model_term_lists=feat_builder.get_top_model_term_lists())
print("Llego aqui")
open("Convention-Visualization-Empath.html", "wb").write(html.encode('utf-8'))