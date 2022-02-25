from nltk.tokenize import word_tokenize
import csv

words = []

def get_data():
    with open('C:\\Users\\LARSI-EQUIPO2\\Desktop\\programas\\Datos\\Tweets.csv', "r") as records:
        for record in csv.reader(records):
            yield record

data = get_data()
next(data)  # skip header

for row in data:
    for sent in row:
        for word in word_tokenize(sent):
            if word not in words:
                words.append(word)
print(words)