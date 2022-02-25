from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

example_sent = """es simplemente el texto de relleno de las imprentas y archivos de texto. 
Lorem Ipsum ha sido el texto de relleno estándar de las industrias desde el año 1500, cuando un 
impresor (N. del T. persona que se dedica a la imprenta) desconocido usó una galería de 
textos y los mezcló de tal manera que logró hacer un libro de textos especimen."""

stop_words = set(stopwords.words('spanish'))

word_tokens = word_tokenize(example_sent)

filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

filtered_sentence = []

for w in word_tokens:
	if w not in stop_words:
		filtered_sentence.append(w)

print(word_tokens)
print('------------------------')
print(filtered_sentence)
