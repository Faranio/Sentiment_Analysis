#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import json
import re
import string
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.ensemble import RandomForestClassifier


def read(file_name):
	path1 = file_name
	data_file1 = open(path1, 'r')
	data1 = data_file1.read()
	json_data = json.loads(data1)
	return json_data

def process_str(this_str):
	this_str = this_str.lower()
	cleanr = re.compile('<.*?>')
	this_str = re.sub(cleanr, '', this_str).replace(' в ', ' ').replace(' а ', ' ').replace(' и ', ' ').replace('\n', ' ').replace('«', ' ').replace('»', ' ').replace('\"', ' ').replace('\t', ' ').replace('–', ' ').replace('–-', ' ').replace('$', 'txttndollar').replace('тенге', 'txttndollar').replace('рубль', 'txttndollar').replace('рублей', 'txttndollar').replace('доллар', 'txttndollar').replace('долларов', 'txttndollar').replace('%', 'txttnperc').replace('процентов', 'txttnperc').replace('процент', 'txttnperc')
	this_str = re.sub(r'http\S+', '', this_str)
	translator = str.maketrans('', '', string.punctuation)
	this_str = this_str.translate(translator)
	this_str = re.sub('(\d+[-/]\d+[-/]\d+)', 'txttndate', this_str)
	this_str = re.sub('\d\d\d\d г', 'txttndate', this_str)
	this_str = re.sub('(\d+[./]\d+[./]\d+)', 'txttndate', this_str)
	this_str = this_str.replace('января', 'txttndate').replace('февраля', 'txttndate').replace('марта', 'txttndate').replace('апреля', 'txttndate').replace('мая', 'txttndate').replace('июня', 'txttndate').replace('июля', 'txttndate').replace('августа', 'txttndate').replace('сентября', 'txttndate').replace('октября', 'txttndate').replace('ноября', 'txttndate').replace('декабря', 'txttndate').replace('январь', 'txttndate').replace('февраль', 'txttndate').replace('март', 'txttndate').replace('апрель', 'txttndate').replace('май', 'txttndate').replace('июнь', 'txttndate').replace('июль', 'txttndate').replace('август', 'txttndate').replace('сентябрь', 'txttndate').replace('октябрь', 'txttndate').replace('ноябрь', 'txttndate').replace('декабрь', 'txttndate')
	ps = PorterStemmer()
	rs = SnowballStemmer("russian")
	l = this_str.split()
	l = [rs.stem(word) for word in l]
	l = [ps.stem(word) for word in l]
	this_str = ' '.join(l)
	while 'txttndate txttndate' in this_str or 'txttndatetxttndate' in this_str:
		this_str = this_str.replace('txttndatetxttndate', 'txttndate').replace('txttndate txttndate', 'txttndate')
	this_str = re.sub('\d', 'txttnnumb', this_str)
	while 'txttnnumbtxttnnumb' in this_str or 'txttnnumb txttnnumb' in this_str:
		this_str = this_str.replace('txttnnumbtxttnnumb', 'txttnnumb').replace('txttnnumb txttnnumb', 'txttnnumb')
	this_str = this_str.replace('txttnnumb txttnperc', 'txttnstat').replace('txttnnumbtxttnperc', 'txttnstat').replace('txttnperc txttnnumb', 'txttnstat').replace('txttnperctxttnnumb', 'txttnstat')
	while '  ' in this_str:
		this_str = this_str.replace('  ', ' ')
	return this_str

def save(this_data, this_file):
	with open(this_file, 'w') as jsonfile:
		json.dump(this_data, jsonfile, ensure_ascii=False)

def predict_hx(X_set, params):
	res = np.dot(X_set, np.transpose(params))
	return res

def random_thetas(length, width):
	return np.random.random_sample((length, width))

def compute_cost(m, X_set, Y_set, params):
	xs = (np.ones(m), X_set)
	hx = predict_hx(np.column_stack(xs), params)
	J = (1/(2*m))*np.sum(np.square(np.subtract(hx, Y_set)))
	return J

def gradient_descent(m, iterations, alpha, X_set, Y_set, params):
	xs = (np.ones(m), X_set)
	for i in range(0, iterations):
		hx = predict_hx(np.column_stack(xs), params)
		params = np.subtract(params, (alpha/m)*np.sum([np.sum(np.subtract(hx, Y_set)) * xx for xx in X_set]))
		print(params, np.sum([np.sum(np.subtract(hx, Y_set)) * xx for xx in X_set]))
		if i%100 == 0:
			plot_line(X_set, Y_set, params)

def text_to_arr(this_text, this_features):
	this_text = this_text.split()
	output = []
	for features in this_features:
		if features in this_text:
			output.append(1)
		else:
			output.append(0)
	return output


# features matrix
read_features = read("top_words.json")
input_features = read_features.keys()
# end of features

# read data
training_data = read("proc_train.json")
texts = [d['text'] for d in training_data]
y = [d['sentiment'] for d in training_data]
index = [d['id'] for d in training_data]

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
train_data_features = vectorizer.fit_transform(texts)
train_data_features = train_data_features.toarray()
vocab = vectorizer.get_feature_names()
dist = np.sum(train_data_features, axis = 0)
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data_features, y)
num_reviews = len(y)
result = forest.predict(train_data_features)
accuracy = 0
for i in range(0, num_reviews):
	if y[i] == result[i]:
		accuracy += 1
	accuracy = accuracy / num_reviews * 100
print("The accuracy is " + str(accuracy) + "%.")
# end of read data

# process text into input
#print("process to input...")
#X = []
#for text_this in texts:
#	X.append(text_to_arr(text_this, input_features))
# end of text into input


#print("training...")
# clf = svm.SVC()
# clf.fit(X, y)
# SVC(C = 1.0, cache_size = 200, class_weight = None, coef0 = 0.0, decision_function_shape = 'ovr', degree = 3, gamma = 'auto', kernel = 'rbf', max_iter = -1, probability = False, random_state = None, shrinking = True, tol = 0.001, verbose = False)
#classifier_linear = svm.SVC(kernel='linear')
#classifier_linear.fit(X, y)


cv_data = read("proc_train.json")
cv_texts = [d['text'] for d in cv_data]
cv_y = [d['sentiment'] for d in cv_data]
cv_index = [d['id'] for d in cv_data]

#("process to input... (cv)")
#cv_X = []
#for text_this in texts:
#	cv_X.append(text_to_arr(text_this, input_features))

#print("predicting")
#for xs in cv_X:
#	print(classifier_linear.predict(xs))

# for i in range(0, len(y)):
# 	if y[i] == "negative":
# 		y[i] = 0
# 	elif y[i] == "positive":
# 		y[i] = 1
# 	else:
# 		y[i] = 0.5




# def train_svm(X, Y, C, kernelFunction):
# 	max_passes = 5
# 	tol = 0.001
# 	m = len(X);
# 	n = len(X[0]);
# 	for i in range(0, len(Y)):
# 		if Y[i] == 0:
# 			Y[i] = -1
# 	alphas = np.zeros(m, 1);
# 	b = 0;
# 	E = np.zeros(m, 1);
# 	passes = 0;
# 	eta = 0;
# 	L = 0;
# 	H = 0;

# 	# 'linearKernel'
# 	K = np.dot(X, np.transpose(X))
# 	dots = 12
# 	while passes < max_passes:
# 		num_changed_alphas = 0
# 		for i in range(1:m):
# 			E[i] = b + np.sum(np.muliply(np.muliply(alphas, Y), [el for el in K][i])) - Y[i]
# 			if (Y[i]*E[i] < -tol and alphas[i] < C) or (Y[i]*E[i] > tol and alphas[i] > 0):
# 				j = math.ceil(m * np.random.rand())
# 				while j == i:
# 					j = math.ceil(m * np.random.rand())
# 				E[j] = b + np.sum(np.muliply(np.muliply(alphas, Y), [el for el in K][j])) - Y[i]
# 				alpha_i_old = alphas[i]
# 				alpha_j_old = alphas[j]
# 				if Y[i] == Y[j]:
# 					L = np.max(alphas(j) + alphas(i) - C)#0, 
# 					H = min(C, alphas(j) + alphas(i))
# 				else:
# 					L = max(0, alphas(j) - alphas(i))
# 					H = min(C, C + alphas(j) - alphas(i))







# find top words
# top_words_pos = {}
# top_words_neg = {}
# top_words_neu = {}

# for i in range(0, len(y)):
# 	this_text = texts[i].split()
# 	for word in this_text:
# 		if word in top_words_neg.keys():
# 			top_words_neg[word] = top_words_neg[word] + 1
# 		else:
# 			top_words_neg[word] = 1

# top_one = {}
# top_two = {}
# top_three = {}

# for key in top_words_neg.keys():
# 	if top_words_neg[key] > 100:
# 		top_one[key] = top_words_neg[key]
# 	if top_words_neg[key] > 200:
# 		top_two[key] = top_words_neg[key]
# 	if top_words_neg[key] > 300:
# 		top_three[key] = top_words_neg[key]

# print(len(top_one), len(top_two), len(top_three))

# save(top_one, "top_words.json")
# save(top_two, "top_words2.json")
# save(top_three, "top_words3.json")
# end of top words



# process data
# for i in range(0, len(y)):
# 	if y[i] == "negative":
# 		y[i] = 0
# 	elif y[i] == "positive":
# 		y[i] = 1
# 	else:
# 		y[i] = 0.5


# dict_data = []

# for s in range(0, int(len(texts)*0.7)):
# 	proc_s = process_str(texts[s])
# 	dict_data.append({'text': proc_s, 'sentiment': y[s], 'id': index[s]})
# 	print("preprocessed: " + str(s))

# save(dict_data, "proc_train.json")

# dict_data = []

# for s in range(int(len(texts)*0.7), len(texts)):
# 	proc_s = process_str(texts[s])
# 	dict_data.append({'text': proc_s, 'sentiment': y[s], 'id': index[s]})
# 	print("preprocessed: " + str(s))

# save(dict_data, "proc_cv.json")
# end of process data
		



