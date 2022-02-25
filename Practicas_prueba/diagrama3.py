import spacy
from scattertext import SampleCorpora, PhraseMachinePhrases, dense_rank, RankDifference, AssociationCompactor, produce_scattertext_explorer
from scattertext.CorpusFromPandas import CorpusFromPandas

corpus = (CorpusFromPandas(SampleCorpora.ConventionData2012.get_data(),
                           category_col='party',
                           text_col='text',
                           feats_from_spacy_doc=PhraseMachinePhrases(),
                           nlp=spacy.load('en_core_web_sm',parser=False))
          .build()).compact(AssociationCompactor(4000))

html = produce_scattertext_explorer(corpus,
                                    category='democrat',
                                    category_name='Democratic',
                                    not_category_name='Republican',
                                    minimum_term_frequency=0,
                                    pmi_threshold_coefficient=0,
                                    transform=dense_rank,
                                    metadata=corpus.get_df()['speaker'],
                                    term_scorer=RankDifference(),
                                    width_in_pixels=1000)

open('PhraseMachine.html','wb').write(html.encode('utf-8'))