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

    Y=lambda df: df["Rosengren's S"],

    Xpos=lambda df: st.Scalers.log_scale(df.X),
    Ypos=lambda df: st.Scalers.scale(df.Y),
)


dispersion_df = dispersion_df.assign(
    Expected=lambda df: KNeighborsRegressor(n_neighbors=10).fit(
        df.X.values.reshape(-1, 1), df.Y
    ).predict(df.X.values.reshape(-1, 1)),
    Residual=lambda df: df.Y - df.Expected,
    ColorScore=lambda df: st.Scalers.stretch_neg1_to_1(df.Residual)
)

html = st.dataframe_scattertext(
    corpus,
    plot_df=dispersion_df,
    metadata=corpus.get_df()['speaker'] + ' (' + corpus.get_df()['party'].str.upper() + ')',
    ignore_categories=True,
    x_label='Log Frequency',
    y_label="Rosengren's S",
    y_axis_labels=['Less Dispersion', 'Medium', 'More Dispersion'],
    color_score_column='ColorScore',
    header_names={'upper': 'Lower than Expected', 'lower': 'More than Expected'},
    left_list_column='Residual',
    background_color='#e5e5e3'
)

open("Customizing-Visualization_color_10.html", 'wb').write(html.encode('utf-8'))