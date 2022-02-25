import pytextrank, spacy
import scattertext as st
import numpy as np
from scattertext import AssociationCompactor, dense_rank, produce_scattertext_explorer, RankDifference

nlp = spacy.load('en_core_web_sm')

convention_df = st.SampleCorpora.ConventionData2012.get_data().assign(
    parse=lambda df: df.text.apply(nlp),
    party=lambda df: df.party.apply({'democrat': 'Democratic', 'republican': 'Republican'}.get)
)
corpus = st.CorpusFromParsedDocuments(
    convention_df,
    category_col='party',
    parsed_col='parse',
    feats_from_spacy_doc=st.PyTextRankPhrases()
).build().compact(
    AssociationCompactor(2000, use_non_text_features=True)
)

term_category_scores = corpus.get_metadata_freq_df('')
print(term_category_scores)

term_ranks = np.argsort(np.argsort(-term_category_scores, axis=0), axis=0) + 1
metadata_descriptions = {
    term: '<br/>' + '<br/>'.join(
        '<b>%s</b> TextRank score rank: %s/%s' % (cat, term_ranks.loc[term, cat], corpus.get_num_metadata())
        for cat in corpus.get_categories())
    for term in corpus.get_metadata()
}

category_specific_prominence = term_category_scores.apply(
    lambda r: r.Democratic if r.Democratic > r.Republican else -r.Republican,
    axis=1
)

html = produce_scattertext_explorer(
    corpus,
    category='Democratic',
    not_category_name='Republican',
    minimum_term_frequency=0,
    pmi_threshold_coefficient=0,
    width_in_pixels=1000,
    transform=dense_rank,
    use_non_text_features=True,
    metadata=corpus.get_df()['speaker'],
    term_scorer=RankDifference(),
    sort_by_dist=False,
    topic_model_term_lists={term: [term] for term in corpus.get_metadata()},
    topic_model_preview_size=0,
    metadata_descriptions=metadata_descriptions,
    use_full_doc=True
)

open("PyTextRankProminenceScore.html", 'wb').write(html.encode('utf-8'))