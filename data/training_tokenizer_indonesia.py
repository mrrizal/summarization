# import punkt
import nltk.tokenize.punkt

# Make a new Tokenizer
tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()

import codecs
text = codecs.open("wikipedia_bahasa_indonesia.txt","Ur","iso-8859-2").read()

# Train tokenizer
tokenizer.train(text)

# Dump pickled tokenizer
import pickle
out = open("indonesia.pickle","wb")
pickle.dump(tokenizer, out)
out.close()
