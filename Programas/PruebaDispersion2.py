import scattertext as st
import pandas as pd
import spacy
from sklearn.neighbors import KNeighborsRegressor

nlp = spacy.load('en_core_web_sm')
df = pd.read_csv('C:\\Users\\Oscar Tello\\Desktop\\Datos\\Tweets.csv')
df = df.assign(
    parse=lambda df: df.tweet.apply(nlp))
corpus = st.CorpusWithoutCategoriesFromParsedDocuments(
    df, parsed_col='parse'
).build().get_unigram_corpus().remove_infrequent_words(minimum_term_count=6)

corpus.get_categories()

dispersion = st.Dispersion(corpus)

dispersion_df = dispersion.get_df()

dispersion_df = dispersion_df.assign(
    X=lambda df: df.Frequency,

    Y=lambda df: df["Rosengren's S"]
)
dispersion_df = dispersion_df.assign(
    Xpos=lambda df: st.Scalers.log_scale(df.X),
    Ypos=lambda df: st.Scalers.scale(df.Y)
)

dispersion_df = dispersion_df.assign(
    Expected=lambda df: KNeighborsRegressor(n_neighbors=10).fit(
        df.X.values.reshape(-1, 1), df.Y
    ).predict(df.X.values.reshape(-1, 1))
)

dispersion_df =dispersion_df.assign(
    Residual=lambda df: df.Y - df.Expected
)

dispersion_df =dispersion_df.assign(
    ColorScore=lambda df: st.Scalers.scale_center_zero_abs(df.Residual)
)

html = st.dataframe_scattertext(
    corpus,
    plot_df=dispersion_df,
    metadata=corpus.get_df()['tweet'],
    ignore_categories=True,
    x_label='Log Frequency',
    y_label="Rosengren's S",
    y_axis_labels=['More Dispersion', 'Medium', 'Less Dispersion'],
    color_score_column='ColorScore',
    header_names={'upper': 'Lower than Expected', 'lower': 'More than Expected'},
    left_list_column='Residual',
    background_color='#b5b3b1'
)

open("prueba_dispersion2.html", 'wb').write(html.encode('utf-8'))