import scattertext as st
import spacy
from sklearn.neighbors import KNeighborsRegressor

nlp = spacy.load('en_core_web_sm')
df = st.SampleCorpora.ConventionData2012.get_data().assign(
    parse=lambda df: df.text.apply(nlp))
corpus = st.CorpusWithoutCategoriesFromParsedDocuments(
    df, parsed_col='parse'
).build().get_unigram_corpus().remove_infrequent_words(minimum_term_count=6)

corpus.get_categories()
dispersion = st.Dispersion(corpus)
dispersion_df = dispersion.get_df()
dispersion_df = dispersion_df.assign(
    X=lambda df: df.Frequency,

    Y=lambda df: df["Frequency"],
)
dispersion_df = dispersion_df.assign(
    Xpos=lambda df: st.Scalers.log_scale(df.X),
    Ypos=lambda df: st.Scalers.scale(df.Y),
)
html = st.dataframe_scattertext(
    corpus,
    plot_df=dispersion_df,
    metadata=corpus.get_df()['speaker'] + ' (' + corpus.get_df()['party'].str.upper() + ')',
    ignore_categories=True,
    x_label='Log Frequency',
    y_label="Frequency",
    y_axis_labels=['Less Dispersion', 'Medium', 'More Dispersion'],
)

open("Customizing-Visualization_Frequency.html", 'wb').write(html.encode('utf-8'))

