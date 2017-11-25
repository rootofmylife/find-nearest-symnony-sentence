from nltk import sent_tokenize

senta = open('./input/trained_data.txt', 'r').read()

sym = r"~`!@#$%^&*()_-+=[]{}|;':\"”“,./<>?"

sent = sent_tokenize(senta)

sent = [s.lower() for s in sent]

sent = [s.translate({ord(c): "" for c in sym}) for s in sent]

text = open('./output/clear_data.txt', 'w')

for s in sent:
    text.write(s + '\n')

text.close()

