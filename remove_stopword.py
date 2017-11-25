from collections import defaultdict
from gensim import corpora, models, similarities

with open('./text_not_remove_stop_word.txt') as f:
    mylist = f.read().splitlines()

with open('./stopwords.txt') as f:
    stop_words = f.read().splitlines()
############
'''Remove [] and replace _'''
'''
mylist = [line.split(' ') for line in mylist]
removed_b_list = []

for s in mylist:
    temp_arr = []
    word_more_than_2 = []
    global c1
    c1 = 0
    global c2
    c2 = 0
    for item in s:
        c1 += item.count('[')
        c2 += item.count(']')
        if ((c1 + c2) == 2) and (len(word_more_than_2) == 0):
            temp_arr.append(item.replace('[','').replace(']',''))
            c1 = c2 = 0
        elif ((c1 + c2) == 2) and (len(word_more_than_2) != 0):
            temp_s = ""
            for i in word_more_than_2:
                temp_s += i + "_"
            temp_s += item
            temp_arr.append(temp_s.replace('[','').replace(']',''))
            c1 = c2 = 0
            word_more_than_2 = []
        elif (c1 + c2) < 2:
            word_more_than_2.append(item)
    removed_b_list.append(temp_arr)

removed_b_list = [' '.join(s) for s in removed_b_list]

removed_b_list = [s.split(' ') for s in removed_b_list]

mylist = removed_b_list
'''
############
texts = [[word for word in doc.split(' ') if word not in stop_words]
         for doc in mylist]

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

'''remove empty list'''
texts = [x for x in texts if x]

#for t in texts:
#    print(t)

dictionary = corpora.Dictionary(texts)
dictionary.save('./word_dict.dict')  # store the dictionary, for future reference

#print(dictionary)

#print(dictionary.token2id)

corpus = [dictionary.doc2bow(text) for text in texts]

corpora.MmCorpus.serialize('./dict/corpus.mm', corpus)

#print(corpus)
