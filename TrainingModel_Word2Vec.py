# Train model
import re, os
import numpy as np
import pandas as pd
from pprint import pprint
# Gensim
import gensim
from gensim import models
import gensim.downloader as api
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.word2vec import Word2Vec
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
# spacy for lemmatization
import spacy
# NLTK Stop words
import nltk
from nltk.corpus import stopwords
# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# TODO change news source before running!!!
newsSource = 'Breitbart'

# A lot is taken from gensim demo at: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
# and https://radimrehurek.com/gensim/auto_examples/core/run_topics_and_transformations.html#transforming-vectors

basePracticeDataFilePath = '/Users/yanisa/GoogleDrive/Publications_Conferences/Code/2020.CovidMetaphorMetonymyBookChptCollab/FinalData/{}/AllFiles_ForTraining/'.format(newsSource)
origDataFile = os.listdir(basePracticeDataFilePath)

# To read each file and split on the \n (so I can only read the text lines)
# Next, separate each article by sentences, and add them to the final list
listOfTexts = []
for eachFile in origDataFile:
    if eachFile == '.DS_Store':
        continue
    else:
        readEachFile = open(basePracticeDataFilePath + eachFile).read()
        fixedFile = readEachFile.split('\n') 
        onlyTextFromFile = fixedFile[6]
        getSentencesFromEachFile = nltk.sent_tokenize(onlyTextFromFile)
        listOfTexts.extend(getSentencesFromEachFile)

# SETTING UP THE CORPUS
# Getting the list of stop words/common words to be excluded
stop_words = set(stopwords.words('english'))
# more can be added w/ stop_words.extend([]) if needed - will have to see data first

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

print('Starting to tokenize...')
data_words = list(sent_to_words(listOfTexts)) 

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=3, threshold=10) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=10)  # higher threshold fewer phrases.

# Faster way to get a sentence as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
#print(trigram_mod[bigram_mod[data_words[0]]]) # can only pass in list of words not slice (list of lists in this case)

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    # https://spacy.io/api/annotation
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model
nlp = spacy.load('en_core_web_sm')

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# INITIALIZE AND TRAIN MODEL
model = Word2Vec(window=5, # window is for max distance between current & predicted word
                min_count=1, # ignores any words that only appear once
                workers=2, # how many cpu cores to use (my laptop has 4)
                sg=1, # so it uses skip-gram (not cbow)
                size=200)  # number of dimensions for each vector (200 were in Dr Liu's paper)
                                                           
model.build_vocab(sentences=texts)
model.train(sentences=texts, total_examples=len(texts), epochs=25)  # train - epochs = # of iterations through the text

# So I don't have to re-train it every time 
model.save("./Word2VecModels/{}word2vec.model".format(newsSource))
# model = Word2VecText.load(fname)