import re, os
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
# Gensim
import gensim
from gensim import models
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.word2vec import Word2Vec
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
from collections import defaultdict

newsSource = 'CNN' # TODO change per source

# A lot is taken from gensim demo at: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
# and https://radimrehurek.com/gensim/auto_examples/core/run_topics_and_transformations.html#transforming-vectors

# TODO make the folder w/ all files before running each news source
basePracticeDataFilePath = '/Users/yanisa/GoogleDrive/Publications_Conferences/Code/2020.CovidMetaphorMetonymyBookChptCollab/FinalData/{}/AllFiles_ForTraining/'.format(newsSource)
origDataFile = os.listdir(basePracticeDataFilePath)

# To read each file and split on the \n (so I can only read the text lines)
listOfTexts = []
for eachFile in origDataFile:
    if eachFile == '.DS_Store':
        continue
    else:
        readEachFile = open(basePracticeDataFilePath + eachFile).read()
        fixedFile = readEachFile.split('\n') 
        onlyTextFromFile = ' '.join(fixedFile[6:]) # Some files have text going into next line for some reason
        getSentencesFromEachFile = nltk.sent_tokenize(onlyTextFromFile)
        listOfTexts.extend(getSentencesFromEachFile)

# Getting the list of stop words/common words to be excluded
stop_words = set(stopwords.words('english'))
# more can be added w/ stop_words.extend([]) if needed - will have to see data first

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(sentence, deacc=True))  # deacc=True removes punctuations

print('Starting to tokenize...')
data_words = list(sent_to_words(listOfTexts)) 

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=3, threshold=10) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=10)  # higher threshold fewer phrases.

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]]) # can only pass in list of words not slice (list of lists in this case)

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

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm')

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# IF WANT TO ADD:
# Remove words that appear less than 5 times 
#   will have to change filenames below if adding!!!
# frequency = defaultdict(int)
# for text in data_lemmatized:
#     for token in text:
#         frequency[token] = frequency[token] + 1

# origDataWords = data_lemmatized[0]
# data_lemmatized = [[token for token in text if frequency[token] > 4] for text in data_lemmatized]
# For debugging - to check list of words that are excluded above
# for word,count in frequency.items():
#     if count <= 4:
#         print(word)
# exit()

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=3, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
#doc_lda = lda_model[corpus]

# Visualize the topics
#pyLDAvis.enable_notebook()
print('Working on creating visualization...')
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, n_jobs=1, R=50) # n_jobs is so it uses up less cpu
print('Going to save html and json...')
pyLDAvis.save_html(vis, 'LDA_Visualization_{}.html'.format(newsSource))
pyLDAvis.save_json(vis, 'LDA_Visualization_{}.json'.format(newsSource))

# TODO NEXT: get keywords from json, save to a .txt file (as temp[NewsSource]), and format
