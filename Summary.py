import nltk.data
from nltk.stem import LancasterStemmer, PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import pandas as pd
from numpy import linalg, sqrt
import math

class  Summary:

	vectorize = CountVectorizer()
	
	def getParagraph(self, text):
		return text.split("\n\n")

	def getSentence(self, paragraph):
		tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')
		sentences = tokenizer.tokenize(paragraph)
		return sentences

	#stemming
	def stemmed_words(self, doc):
		stemmer = PorterStemmer()
		return (stemmer.stem(w) for w in self.stopWordEnglish(doc))

	#remove stop word (english stop word)
	def stopWordEnglish(self, doc):
		analyzer = CountVectorizer().build_analyzer()
		english_stop = set(stopwords.words('english'))
		return (word for word in analyzer(doc) if word not in english_stop)
	
	'''
	create document term-matrix
	bisa menggunakan 3 cara untuk membuat / memenuhi document term-matrix
	1. term frekwensi
	2. binary (0/1)
	3. tf-idf
	'''
	def getDTM(self, sentences, binaryMode=False, mode='CountVectorizer'):
		if(mode=='CountVectorizer'):
			self.vectorize = CountVectorizer(min_df=0.0, analyzer=self.stemmed_words, binary=binaryMode, max_df=1.0)
		elif(mode=='TfidfVectorizer'):
			self.vectorize = TfidfVectorizer(min_df=0.0, analyzer=self.stemmed_words, max_df=1.0)
		
		dtm = self.vectorize.fit_transform(sentences)
		#print(pd.DataFrame(dtm.toarray(), index=sentences, columns=self.vectorize.get_feature_names()))
		#print()
		return dtm
		
	#create svd	
	def getSVD(self, documentTermMatrix, sentences):
		lsa = TruncatedSVD(len(sentences), algorithm='randomized')
		dtm_lsa = lsa.fit_transform(documentTermMatrix)
		dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
		u = lsa.components_
		sigma = lsa.explained_variance_ratio_
		vt = dtm_lsa.T
		return u, sigma, vt


	'''
	sentences selection
	bisa menggunakan 3 metode:
	1. GongLiu 
	2. SteinbergerJezek
	3. cross

	aspect ration digunakan untuk mengatur berapa banyak kalimat di ambil (dalam %)
	contoh 50 -> berarti 50% kalimat diambil

	lebih jelas mengenai senten selection bisa dilihat di 
	https://www.researchgate.net/publication/220195824_Text_summarization_using_Latent_Semantic_Analysis

	'''
	def getSummary(self, u=0, sigma=0, vt=0, approach='GongLiu', aspectRatio=50):
		value = {}
				
		self.aspectRatio = int((aspectRatio/100)*len(vt))

		if(len(vt)>3):
			#metode milik Gong dan Liu
			if approach == 'GongLiu':
				#banyaknya kalimat yang di ambil
				for i in range(self.aspectRatio): 
					maxvalue = 0
					index = 0
					index1 = 0
					for i in range(len(vt)):
						for j in range(len(vt[i])):
							if vt[i][j] > maxvalue:
								maxvalue = vt[i][j]
								index = i
								index1 = j
					vt[index][index1] = 0
					value[index1] = maxvalue

			#metode milik Steinberger dan Jezek
			elif approach == 'SteinbergerJezek':
				for i in range(len(vt)):
					vt[i].append(sqrt(sum([vt[i][j]*sigma[i] for j in range(len(vt[i]))])))

				#print(pd.DataFrame(vt, index=[i+1 for i in range(len(vt))], columns=[i+1 for i in range(len(vt)+1)]))

				data = []
				for i in range(len(vt)):
					data.append(vt[i][len(vt[i])-1])
					
				#banyaknya kalimat yang di ambil
				for i in range(self.aspectRatio): 
					value[data.index(max(data))] = max(data)
					data[data.index(max(data))] = 0


			elif approach == 'SteinbergerJezek2':
				MIN_DIMENSIONS = 3
				REDUCTION_RATIO = 1/1
				dimensions = max(MIN_DIMENSIONS,int(len(sigma)*REDUCTION_RATIO))
				powered_sigma = tuple(s**2 if i < dimensions else 0.0 for i, s in enumerate(sigma))
				data = []

				for column_vector in vt:
					tmp = sum(s*v**2 for s, v in zip(powered_sigma, column_vector))
					data.append(math.sqrt(tmp))
				
				for i in range(self.aspectRatio): 
					value[data.index(max(data))] = max(data)
					data[data.index(max(data))] = 0

			#metode cross 
			elif approach == 'cross':
				lengthOfSentences = [0 for i in range(len(vt))]
				for i in range(len(vt)):
					average = sum(vt[i])/len(vt[i])
					for j in range(len(vt[i])):
						if vt[i][j] < average:
							vt[i][j] = 0
					vt[i].append(average)
					tmp = vt[i][:-1]
					lengthOfSentences = [sum(y) for y in zip([tmp[x]*sigma[x] for x in range(len(tmp))], lengthOfSentences)]
					#print(vt[i])

				vt.append(lengthOfSentences)
				#for i in range(len(vt)):
				#	print(vt[i])
				data = vt[len(vt)-1]

				#banyak kalimat yang di pilih
				for i in range(self.aspectRatio):
					value[data.index(max(data))] = max(data)
					data[data.index(max(data))] = 0

		else:
			value[0] = 0
			value[1] = 0

		return value

	def getEvaluation(self, ue, uf):
		evaluation = []
		for i in range(len(ue)):
			try:
				evaluation.append(ue[i]*uf[i])
			except IndexError:
				evaluation.append(0)

		return math.acos(sum(evaluation))
	
		
#main program
def main():
	file = open('sample_plain_text_3.txt', 'r')
	data = file.read()

	summary = Summary()
	paragraph = summary.getParagraph(data)

	for i in paragraph:
		sentences = summary.getSentence(i)
		dtm = summary.getDTM(sentences, mode='CountVectorizer')
		u, sigma, vt = summary.getSVD(dtm, sentences)
		
		#menampilkan hasil summary
		keys = summary.getSummary(sigma=sigma, vt=vt.tolist(), approach='SteinbergerJezek2').keys()
		keys = sorted(keys)
		summaryResult = []
		for key in keys:
			try:
				summaryResult.append(sentences[key])
			except:
				pass

		for i in summaryResult:
			print(i)

		'''
		menampilkan hasil evaluasi(lsa-based)
		dtmSummary = summary.getDTM(summaryResult, mode='CountVectorizer', binaryMode=True)
		uSummary, sigmaSummary, vtSummary = summary.getSVD(dtmSummary, summaryResult)
		evaluation = summary.getEvaluation(sorted(u[0]), sorted(uSummary[0]))
		print("Evaluation : ", evaluation) 
		'''
		
		print()

if __name__ == "__main__":
	main()	

