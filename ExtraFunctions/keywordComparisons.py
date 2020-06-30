# Compare between two lists of keywords which are shared and which only belong to one of the news sources

sharedKeywords = []
firstKeywords = [] # TODO change names if news source changes
secondKeywords = []

firstNewsSource = 'Slate'
secondNewsSource = 'Breitbart' # TODO change if news source/keywords change
numKeywords = 200

firstNewsSourceFile = open('/Users/yanisa/GoogleDrive/Publications_Conferences/Code/2020.CovidMetaphorMetonymyBookChptCollab/DataAnalysis/ListsOfKeywords/{}_{}keywords.txt'.format(firstNewsSource, numKeywords)).read()
secondNewsSourceFile = open('/Users/yanisa/GoogleDrive/Publications_Conferences/Code/2020.CovidMetaphorMetonymyBookChptCollab/DataAnalysis/ListsOfKeywords/{}_{}keywords.txt'.format(secondNewsSource, numKeywords)).read()

firstNewsSourceList = firstNewsSourceFile.split('\n')
secondNewsSourceList = secondNewsSourceFile.split('\n')
print('num first list words: ' + str(len(firstNewsSourceList)))
print('num second list words: ' + str(len(secondNewsSourceList)))
for word in firstNewsSourceList:
    if word in secondNewsSourceList:
        sharedKeywords.append(word)
        secondNewsSourceList.remove(word)
    else:
        firstKeywords.append(word)

print('Shared words: ' + str(sharedKeywords) + '\n')
print('Slate words: ' + str(firstKeywords) + '\n') # TODO change string w/ correct news source
print('Breitbart words: ' + str(secondNewsSourceList))
print(len(sharedKeywords) + len(firstKeywords) + len(secondNewsSourceList))
print(len(sharedKeywords))