import scattertext as st

corpus = (st.CorpusFromPandas(st.SampleCorpora.ConventionData2012.get_data(),
                              category_col='party',
                              text_col='text',
                              nlp=st.whitespace_nlp_with_sentences).build()
          .get_unigram_corpus()
          .compact(st.ClassPercentageCompactor(term_count=2,
                                               term_ranker=st.OncePerDocFrequencyRanker)))

html = st.produce_characteristic_explorer(
    corpus,
    category='democrat',
    category_name='Democratic',
    not_category_name='Reúblican',
    metadata=corpus.get_df()['speaker']
)

open('Demo_Chracteristic_Chart.html','wb').write(html.encode('utf-8'))