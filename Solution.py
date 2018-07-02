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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from Neural_Network_Model import *

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

def text_to_arr(this_text, this_features):
	this_text = this_text.split()
	output = np.array([])
	for features in this_features:
		if features in this_text:
			output = np.append(output, 1)
		else:
			output = np.append(output, 0)
	return output

def accuracy(X, y):

	output = 0
	for i in range(1, X.size):
		if X[i] == y[i]:
			output += 1
	output = output / X.size * 100
	return output

# Input features
read_features = read("top_words.json")
input_features = read_features.keys()
# End

# 0.7 of Training data
training_data = read("proc_train.json")
texts = [d['text'] for d in training_data]
y = [d['sentiment'] for d in training_data]
index = [d['id'] for d in training_data]
# End

# Process text into input
print("Process to input...")
X = []
i = 0
for text_this in texts:
	if i == 10:
		break
	i += 1
	X.append(text_to_arr(text_this, input_features))
X_np = np.array(X) #5784x3247 matrix
# end of text into input

print("Training Neural Network...")

# Neural Network model
y_temp = np.array([y[0:10]]) #1x5784 matrix
layers_dims = [len(input_features), 10, 1] #3247, 10, 3
train_x = X_np.reshape(X_np.shape[0], -1).T #3247x5784 matrix
parameters = L_layer_model(train_x, y_temp, layers_dims, num_iterations = 2500, print_cost = True)
pred_train = predict(train_x, y, parameters)

# Cross-validation Testing
print("Process to input... (CV)")
cv_data = read("proc_cv.json")
cv_texts = [d['text'] for d in cv_data]
cv_y = [d['sentiment'] for d in cv_data]
cv_index = [d['id'] for d in cv_data]

cv_X = []
for text_this in cv_texts:
	cv_X.append(text_to_arr(text_this, input_features))
cv_X_np = np.array(cv_X)

print("Predicting... (CV)")
cv_y_temp = np.array([cv_y])
test_x = cv_X_np.reshape(cv_X_np.shape[0], -1).T
pred_cv = predict(test_x, cv_y, parameters)
