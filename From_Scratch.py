#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import ExtraTreesClassifier

train = pd.read_json("train.json")

train_X = train[0:7000]
cv_X = train[7001:]
train_size = train_X["text"].size
cv_size = cv_X["text"].size

badwords = [
u'я', u'а', u'да', u'но', u'тебе', u'мне', u'ты', u'и', u'у', u'на', u'ща', u'ага',
u'так', u'там', u'какие', u'который', u'какая', u'туда', u'давай', u'короче', u'кажется', u'вообще',
u'ну', u'не', u'чет', u'неа', u'свои', u'наше', u'хотя', u'такое', u'например', u'кароч', u'как-то',
u'нам', u'хм', u'всем', u'нет', u'да', u'оно', u'своем', u'про', u'вы', u'м', u'тд',
u'вся', u'кто-то', u'что-то', u'вам', u'это', u'эта', u'эти', u'этот', u'прям', u'либо', u'как', u'мы',
u'просто', u'блин', u'очень', u'самые', u'твоем', u'ваша', u'кстати', u'вроде', u'типа', u'пока', u'ок'
]

def review_to_words(raw_review):
	review_text = BeautifulSoup(raw_review).get_text()
	letters_only = re.sub(ur'[^а-яА-Я]', " ", review_text)
	letters_only = re.sub(r'http\S+', '', letters_only)
	words = letters_only.lower().split()
	meaningful_words = [w for w in words if not w in badwords]
	return(" ".join(meaningful_words))

print "Cleaning and parsing the training set movie reviews...\n"

clean_train_reviews = []

for i in xrange(0, train_size):
	if((i + 1) % 1000 == 0):
		print "Review %d of %d\n" % (i + 1, train_size)
	clean_train_reviews.append(review_to_words(train["text"][i]))

print "Creating the bag of words...\n"
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()	
vocab = vectorizer.get_feature_names()	
dist = np.sum(train_data_features, axis = 0)
for tag, count in zip(vocab, dist):
	print count, tag

print "Training the Extra Trees Classifier..."

forest = ExtraTreesClassifier(n_estimators = 200)
forest = forest.fit(train_data_features, train["sentiment"][0:train_size])

train_result = forest.predict(train_data_features)

accuracy = 0
for i in range(0, train_size):
	if train_result[i] == train["sentiment"][i]:
		accuracy = accuracy + 1
accuracy = accuracy * 100 / train_size

print "Cleaning and parsing the CV set movie reviews...\n"

clean_cv_reviews = []

for i in xrange(train_size, train_size + cv_size):
	if((i + 1) % 1000 == 0):
		print "Review %d of %d\n" % (i + 1, train_size + cv_size)
	clean_cv_reviews.append(review_to_words(train["text"][i]))

print "Creating the bag of words for CV...\n"
cv_data_features = vectorizer.transform(clean_cv_reviews)
cv_data_features = cv_data_features.toarray()	

print "Training the random forest on CV..."

cv_result = forest.predict(cv_data_features)

cv_accuracy = 0
for i in range(0, cv_size):
	if cv_result[i] == train["sentiment"][i + train_size]:
		cv_accuracy = cv_accuracy + 1
cv_accuracy = cv_accuracy * 100 / (cv_size)

print("The training accuracy is " + str(accuracy) + "%.")
print("The CV accuracy is " + str(cv_accuracy) + "%.")