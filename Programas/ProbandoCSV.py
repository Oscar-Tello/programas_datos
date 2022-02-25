import scattertext as st
import spacy
import pandas as pd
from pprint import pprint
import seaborn as sbn


df = pd.read_csv('C:\\Users\\Oscar Tello\\Desktop\\MCA\\MaterialesTesis\\Tweets.csv')
nlp = spacy.load('es_core_news_sm')

df['opinion'] = df['tweet'].replace(
    {'#100mil':'Negativas','fachos':'Positivas'}
)

df_opiniones = df[df['opinion'].str.contains('fachos|resentido|chayotero', na=False)]

corpus = st.CorpusFromPandas(
    df,
    category_col='opinion',
    text_col='tweet',
    nlp=nlp
).build()
corpus_df = df_opiniones
corpus.get_term_freq_df().to_csv('frecuencia_tweets.csv')

term_freq = corpus.get_term_freq_df()

html = st.produce_scattertext_explorer(
    corpus,
    category='Negativas',
    category_name='Negativas',
    not_category_name='Positivas',
    width_in_pixels=1000,
    metadata=df['tweet']
)


#opinion_positiva= term_freq(df_opiniones)

#term_freq['Opiniones Positivas'] = corpus.get_scaled_f_scores('fachos')
#print(term_freq['Opiniones Positivas'])
#pprint(list(term_freq.sort_values(by='opiniones', ascending=False).index[:10]))
