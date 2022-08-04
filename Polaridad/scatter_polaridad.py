# -*- coding: utf-8 -*-
import scattertext as st
import pandas as pd

#df = pd.read_csv('C:\\Users\\LARSI-EQUIPO2\\Desktop\\programas\\Datos\\sentimiento.csv')
df = pd.read_csv('C:\\Users\\Tello\\Desktop\\programas\\programas_datos\\Datos\\sentimiento.csv')
#movie_df.to_csv("datos_tomatoes.csv")

df.sentimiento = df.sentimiento.apply\
	(lambda x: {'negativo': 'Negativo', 'positivo': 'Positivo', 'neutro': 'Neutro'}[x])
corpus = st.CorpusFromPandas(
	df,
	category_col='sentimiento',
	text_col='tweet',
	nlp=st.whitespace_nlp_with_sentences
).build().get_unigram_corpus()

semiotic_square = st.SemioticSquare(
	corpus,
	category_a='Positivo',
	category_b='Negativo',
	neutral_categories=['Neutro'],
	scorer=st.RankDifference(),
	labels={'not_a_and_not_b': 'Plot Descriptions', 'a_and_b': 'Reviews'}
)

html = st.produce_semiotic_square_explorer(semiotic_square,
                                           category_name='Positivo',
                                           not_category_name='Negativo',
                                           x_label='Puntaje negativo',
                                           y_label='Puntaje neutro',
                                           neutral_category_name='Description',
                                           metadata=df['date'])

open("WWIII_polaridad.html", 'wb').write(html.encode('utf-8'))