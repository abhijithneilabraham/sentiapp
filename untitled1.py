#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 22:34:48 2018

@author: abhijith
"""
from string import punctuation
from keras.models import model_from_json
from os import listdir
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
def load_model():
    
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def doc_to_clean_lines(doc, vocab):
	clean_lines = list()
	lines = doc.splitlines()
	for line in lines:
		# split into tokens by white space
		tokens = line.split()
		# remove punctuation from each token
		table = str.maketrans('', '', punctuation)
		tokens = [w.translate(table) for w in tokens]
		# filter out tokens not in vocab
		tokens = [w for w in tokens if w in vocab]
		clean_lines.append(tokens)
	return clean_lines

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		doc = load_doc(path)
		doc_lines = doc_to_clean_lines(doc, vocab)
		# add lines to list
		lines += doc_lines
	return lines

vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
doc='nn.txt'
x=doc_to_clean_lines('doc',vocab)
print(x)