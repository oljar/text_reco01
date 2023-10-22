from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import codecs



import numpy as np

import re

# otwieranie pliku
texts = []
labels = []
with codecs.open('teach_list.txt', 'r','UTF-8')as f:
    teach_lines = f.read()

    teach_list = teach_lines.split('\n')

    for ln in teach_list:
        if ln:
            text,label = ln.split('=')

            text,label = text.replace('\r','').strip(),label.replace('\r','').strip()

            texts.append(text)
            labels.append(label)

print (texts) # przykładowe dane tekstowe
print (labels) # etykiety sentymentu dla danych tekstowych



# tworzenie wektorów cech dla danych tekstowych i trenowanie klasyfikatora
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
clf = MultinomialNB()
X_clf = clf.fit(X, labels)


predicted_labels = []

prec = round((1/len(texts)),8)

# odczytywanie zawartości pliku

with codecs.open('in.txt', 'r','UTF-8')as f:
    zawartosc = f.read()

# dzielenie zawartości na zdania
zdania = zawartosc.replace('\n','').strip().split('.')


# wyświetlanie zdań
for zdanie in zdania:
    X_new = vectorizer.transform([zdanie])
    predicted_proba = clf.predict_proba(X_new)

    # print(clf.predict_proba(X_new))
    max_proba = np.max(predicted_proba)
    # if max_proba <= 0.3 :  # możesz dostosować próg do swoich potrzeb

    if max_proba <= prec and zdanie:  # możesz dostosować próg do swoich potrzeb
        predicted_labels.append('brak')
        print(zdanie,[predicted_labels[-1]])


    elif zdanie:
        predicted_label = clf.predict(X_new)
        predicted_labels.append(predicted_label[0])
        print (zdanie , predicted_label)


with open('out.txt', 'w', encoding='utf-8') as f:
            for n in range(len(zdania)):

                if zdania[n]:
                    f.write(f"{zdania[n]} = {predicted_labels[n]}"+'\n')


