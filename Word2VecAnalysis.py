import gensim.downloader as api
from gensim.models.word2vec import Word2Vec

newsSource = 'Breitbart'
numKeywords = 200 # TODO will have to change these to make appropriate

# Load model
model = Word2Vec.load('/Users/yanisa/GoogleDrive/Publications_Conferences/Code/2020.CovidMetaphorMetonymyBookChptCollab/DataAnalysis/Word2VecModels/{}word2vec.model'.format(newsSource))

# Import list of keywords
topKeywords = open('/Users/yanisa/GoogleDrive/Publications_Conferences/Code/2020.CovidMetaphorMetonymyBookChptCollab/DataAnalysis/ListsOfKeywords/{}_{}keywords.txt'.format(newsSource, numKeywords)).read()

# make the keywords into a list to be looped over later
listOfTopKeywords = topKeywords.split('\n')

# loop over the list to get each word's companion words, then write to a file as it loops around
# TODO make sure file exists
saveCompanionWordsList = open('/Users/yanisa/GoogleDrive/Publications_Conferences/Code/2020.CovidMetaphorMetonymyBookChptCollab/DataAnalysis/ListsOfKeywords/{}_{}KeywordWithCompanionWords.txt'.format(newsSource, numKeywords), 'w')
counter = 1
for keyword in listOfTopKeywords:
    words = model.wv.most_similar(positive=keyword, 
                                  topn=10) # Dr Liu's other paper had 8
    print(str(counter) + '. ' + keyword + ': ' + str(words))
    saveCompanionWordsList.write(str(counter) + '. ' + keyword + ': ' + str(words) + '\n\n')
    counter = counter + 1

# IF WANT TO LOOK AT ONE SPECIFIC WORD -
#   comment out the above section (except model and import)
#   un-comment out this part and run per word
# specificWord = 'hoax'
# word = model.wv.most_similar(positive=[specificWord],topn=10)
# print(specificWord + ': ' + str(word))