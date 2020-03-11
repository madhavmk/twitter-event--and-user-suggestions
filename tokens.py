# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 14:17:55 2018

@author: SZenkar
"""

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import codecs
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
import pickle
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import CMUTweetTagger
import codecs
from collections import Counter
from datetime import datetime
import fastcluster
from itertools import cycle
import json
import nltk
import numpy as np
import re
#import requests
import os
import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import metrics
#from stemming.porter2 import stem
import string
import sys
import time
import ast      #abstarct syntax trees


writefile=codecs.open("tokenized_text.json",'w',encoding="utf-8")

def tokenize_text(text):
    stop_words = set(stopwords.words('english'))
    text = normalize_text(text)
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words and '[' not in w and '\\' not in w and len(w) > 1 and w.isalpha()]
    return(filtered_sentence)    

def normalize_text(text):
    text.encode('utf-8')
    text=re.sub('@[^\s]+','', text)
    text=re.sub('#[^\s]+', '', text)
    text=re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))','', text)
   # print("bastard got no problems now")
    text=re.sub('&[^\s]+','',text)
    text=re.sub('\n',' ',text)
    text = text.replace(".", ' ')
    text = text.replace("!", ' ')
    text = text.replace("?", ' ')
    text = text.replace("'", ' ')
    text = text.replace("\"", ' ') 
    #text =re.sub('\x[^\s]+',' ',text)
    return(text)

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')
en_stop.extend(["rt","n","t","http","https","co",])

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    
read_file = codecs.open("extract.data","r","utf-8")

# compile sample documents into a list
doc_set = [ ast.literal_eval(i)[0] for i in read_file.readlines() ]

# list for tokenized documents in loop
texts = []
corpus = []
dfVocTimeWindows = {}
tweet_unixtime_old = -1
tid_to_raw_tweet = {}

tid_to_urls_window_corpus = {}
tids_window_corpus = []

t = 0
ntweets = 0
ct = 0 
# loop through document list
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    
    raw = normalize_text(raw)
    
    tokens = tokenize_text(raw)
    
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop and len(i)>1]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    #print(tokens)
    s = ""
    writefile.write("%s\n" % tokens)
    for i in tokens :
        s += i+","
    s = s[:-1]
    corpus.append(s)
    ct += 1
    #if ct > 8000 :
     #   break
#print(corpus)



# tokenizer for CountVectorizer
def custom_tokenize_text(text):
    
	REGEX = re.compile(r",\s*")
	tokens = []
	#print( REGEX.split(text))
	for tok in REGEX.split(text):
		#if "@" not in tok and "#" not in tok:
		if "@" not in tok:
			#tokens.append(stem(tok.strip().lower()))
			tokens.append(tok.strip().lower())
		
	return tokens
	
vectorizer = CountVectorizer(tokenizer=custom_tokenize_text,min_df=max(int(len(corpus)*0.0025), 10), binary=True, ngram_range=(1,3))
X = vectorizer.fit_transform(corpus)  # tweet term matrix
map_index_after_cleaning = {}
Xclean = np.zeros((1, X.shape[1]))
print( "X.shape : ", X.shape[0] )
print ("keep sample with size at least 5 key terms")
for i in range(0, X.shape[0]):
	
	if X[i].sum() > 4:
		#print("X[i].sum() is greater than 4")
		Xclean = np.vstack([Xclean, X[i].toarray()])
		map_index_after_cleaning[Xclean.shape[0] - 2] = i
Xclean = Xclean[1:]
print ("\n\n\n", Xclean , map_index_after_cleaning)
print ("total tweets in window:", ct )
print ("X.shape:", X.shape)
print ("Xclean.shape:", Xclean.shape)
print (map_index_after_cleaning)
#play with scaling of X
X = Xclean
Xdense = np.matrix(X).astype('float')
X_scaled = preprocessing.scale(Xdense)
X_normalized = preprocessing.normalize(X_scaled, norm='l2')
vocX = vectorizer.get_feature_names()
#print( "\n Xdense : \n" , Xdense , "\n X_scaled : \n" , X_scaled , "\n X_normalized : \n",  X_normalized , "\n vocX : " , vocX )

boost_entity = {}

pos_tokens = CMUTweetTagger.runtagger_parse([term.upper() for term in vocX])
print(pos_tokens)
#print "detect entities", pos_tokens
for l in pos_tokens:
	term =''
	for gr in range(0, len(l)):
		term += l[gr][0].lower() + " "
		if "^" in str(l):
			boost_entity[term.strip()] = 2.5
		else: 	 		
	 		boost_entity[term.strip()] = 1.0

dfX = X.sum(axis=0)
#print "dfX:", dfX
dfVoc = {}
wdfVoc = {}
boosted_wdfVoc = {}	
keys = vocX
vals = dfX
for k,v in zip(keys, vals):
	dfVoc[k] = v
	for k in dfVoc: 
		try:
			dfVocTimeWindows[k] += dfVoc[k]
			avgdfVoc = (dfVocTimeWindows[k] - dfVoc[k])/(t - 1)
			#avgdfVoc = (dfVocTimeWindows[k] - dfVoc[k])
		except:
			dfVocTimeWindows[k] = dfVoc[k]
			avgdfVoc = 0
			wdfVoc[k] = (dfVoc[k] + 1) / (np.log(avgdfVoc + 1) + 1)
			try:
				boosted_wdfVoc[k] = wdfVoc[k] * boost_entity[k]
			except: 
				boosted_wdfVoc[k] = wdfVoc[k]
# 					try:
# 						print "\ndfVoc:", k.decode('utf-8'), dfVoc[k]
# 						print "dfVocTimeWindows:", k.decode('utf-8'), dfVocTimeWindows[k]
# 						print "avgdfVoc:", k.decode('utf-8'), avgdfVoc
# 						print "np.log(avgdfVoc + 1):", k.decode('utf-8'), np.log(avgdfVoc + 1)
# 						print "wdfVoc:", k.decode('utf-8'), wdfVoc[k]
# 						print "wdfVoc*boost_entity:", k.decode('utf-8'), wdfVoc[k] * boost_entity[k]
#  					except: pass

# 				print "total VocTimeWindows so far:", len(dfVocTimeWindows)
print("sorted wdfVoc*boost_entity:")

#print(sorted( ((v,k) for k,v in boosted_wdfVoc.items()), reverse=True))

distMatrix = pairwise_distances(X_normalized, metric='cosine')

				#print distMatrix
				#distMatrixXt = pairwise_distances(Xt)
				#print "distMatrixXt.shape", distMatrixXt.shape
				#cluster tweets
print("fastcluster, average, cosine")
L = fastcluster.linkage(distMatrix, method='average')

				#for dt in [0.3, 0.4, 0.5, 0.6, 0.7]:
				#for dt in [0.5]:
dt = 0.85
print("hclust cut threshold:", dt)
#				indL = sch.fcluster(L, dt, 'distance')
indL = sch.fcluster(L, dt*distMatrix.max(), 'distance')
				#print "indL:", indL
freqTwCl = Counter(indL)
print("n_clusters:", len(freqTwCl))
print(freqTwCl)
		
npindL = np.array(indL)
#				print "top50 most populated clusters, down to size", max(10, int(X.shape[0]*0.0025))
freq_th = max(10, int(X.shape[0]*0.0025))
cluster_score = {}
for clfreq in freqTwCl.most_common(50):
    cl = clfreq[0]
    freq = clfreq[1]
    cluster_score[cl] = 0
    if freq >= freq_th:
     		#print "\n(cluster, freq):", clfreq
    		clidx = (npindL == cl).nonzero()[0].tolist()
    		cluster_centroid = X[clidx].sum(axis=0)
    		#print "centroid_array:", cluster_centroid
    		try:
    		#orig_tweet = window_corpus[map_index_after_cleaning[i]].decode("utf-8")
    			cluster_tweet = vectorizer.inverse_transform(cluster_centroid)
    			#print orig_tweet, cluster_tweet, urls_window_corpus[map_index_after_cleaning[i]]
    			#print orig_tweet
    							#print "centroid_tweet:", cluster_tweet
    			for term in np.nditer(cluster_tweet):
    				#print "term:", term#, wdfVoc[term]
    				try:
    					cluster_score[cl] = max(cluster_score[cl], boosted_wdfVoc[str(term).strip()])
    					#cluster_score[cl] += wdfVoc[str(term).strip()] * boost_entity[str(term)] #* boost_term_in_article[str(term)]
    					#cluster_score[cl] = max(cluster_score[cl], wdfVoc[str(term).strip()] * boost_term_in_article[str(term)])
    					#cluster_score[cl] = max(cluster_score[cl], wdfVoc[str(term).strip()] * boost_entity[str(term)])	
    					#cluster_score[cl] = max(cluster_score[cl], wdfVoc[str(term).strip()] * boost_entity[str(term)] * boost_term_in_article[str(term)])
    				except: pass 			
    		except: pass
    		cluster_score[cl] /= freq
    else: break
			
sorted_clusters = sorted( ((v,k) for k,v in cluster_score.items()), reverse=True)
print("sorted cluster_score:")
print(sorted_clusters)
print(len(sorted_clusters))

ntopics = 20
headline_corpus = []
orig_headline_corpus = []
headline_to_cluster = {}
headline_to_tid = {}
cluster_to_tids = {}
for score,cl in sorted_clusters[:ntopics]:
	#print "\n(cluster, freq):", cl, freqTwCl[cl]
	clidx = (npindL == cl).nonzero()[0].tolist()
	#cluster_centroid = X[clidx].sum(axis=0)
	#centroid_tweet = vectorizer.inverse_transform(cluster_centroid)
	#random.seed(0)
	#sample_tweets = random.sample(clidx, 3)
	#keywords = vectorizer.inverse_transform(cluster_centroid.tolist())
	first_idx = map_index_after_cleaning[clidx[0]]
	keywords = corpus[first_idx]
	orig_headline_corpus.append(keywords)
	headline = ''
	print(keywords.split(","))
	for k in keywords.split(","):
		if not '@' in k and not '#' in k:
			headline += k + ","
			headline_corpus.append(headline[:-1])
			headline_to_cluster[headline[:-1]] = cl
			#print( len( headline_to_tid) )
			#headline_to_tid[headline[:-1]] = tids_window_corpus[first_idx]
			meta_tweet = ''
# 						for term in np.nditer(centroid_tweet):
# 							meta_tweet += str(term) + ","
#  						headline_corpus.append(meta_tweet[:-1])
			tids = []
			'''for i in clidx:
				idx = map_index_after_cleaning[i]
				tids.append(tids_window_corpus[idx])
#   						try:
#   							print window_corpus[map_index_after_cleaning[i]]
#   						except: pass	
				cluster_to_tids[cl] = tids	
#    					try:
# # #  						print vectorizer.inverse_transform(X[clidx[0]])
#    						print keywords
# # # 						print tid_to_raw_tweet[tids_window_corpus[first_idx]]
# # # # # 							#print meta_tweet
# # # # # 								#print "[", headline, "\t", keywords, "\t", tids, "\t", turls, "]"		
# # # # # 								#print tweet_time_window_corpus[idx],
'''
read_file.close()
writefile.close()
