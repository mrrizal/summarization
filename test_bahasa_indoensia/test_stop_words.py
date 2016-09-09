f = open('../data/stopwords_id.txt')
stopwords = [word.rstrip('\n') for word in f.readlines()]
print(stopwords)