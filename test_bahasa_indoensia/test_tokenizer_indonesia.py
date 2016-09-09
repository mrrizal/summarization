import nltk.data

def getSentence(data):
	tokenizer = nltk.data.load('tokenizers/punkt/PY3/indonesia.pickle')
	sentence = tokenizer.tokenize(data)
	return sentence

def main():
	f = open('../data/teks_indonesia.txt', 'r')
	data = f.read()
	sentences = getSentence(data)
	for sentence in sentences:
		print(sentence)

if __name__ == '__main__':
	main()