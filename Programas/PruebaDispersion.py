import scattertext as st
import pandas as pd
import spacy

nlp = spacy.load('en_core_web_sm')
df = pd.read_csv('C:\\Users\\LARSI-EQUIPO2\\Desktop\\programas\\Datos\\Tweets.csv')
df = df.assign(
    parse=lambda df: df.tweet.apply(nlp))
#print(df)

corpus = st.CorpusWithoutCategoriesFromParsedDocuments(
    df, parsed_col='parse'
).build().get_unigram_corpus().remove_infrequent_words(minimum_term_count=6)

print('-------------')
print(type(df))
df.parse.to_csv('example3.csv',index=False)

corpus.get_categories()

dispersion = st.Dispersion(corpus)

dispersion_df = dispersion.get_df()

dispersion_df = dispersion_df.assign(
    X=lambda df: df.Frequency,

    Y=lambda df: df["Rosengren's S"],
)
dispersion_df = dispersion_df.assign(
    Xpos=lambda df: st.Scalers.log_scale(df.X),
    Ypos=lambda df: st.Scalers.scale(df.Y),
)

html = st.dataframe_scattertext(
    corpus,
    plot_df=dispersion_df,
    metadata=corpus.get_df()['tweet'],
    ignore_categories=True,
    x_label='Log Frequency',
    y_label="Rosengren's S",
    y_axis_labels=['More Dispersion', 'Medium', 'Less Dispersion'],
)

open("prueba_dispersion.html", 'wb').write(html.encode('utf-8'))