# Remove duplicate keywords and only end up with _____ # of unique keywords
# Files first saved as temp[NEWSSOURCE], then in here the final files are made 

newsSource = 'Slate' # TODO change as necessary
numDesiredKeywords = 200

allWordsFile = open('/Users/yanisa/GoogleDrive/Publications_Conferences/Code/2020.CovidMetaphorMetonymyBookChptCollab/DataAnalysis/ListsOfKeywords/temp{}.txt'.format(newsSource)).read()

allWordsList = allWordsFile.split('\n')
finalListKeywords = []
counter = 1
for word in allWordsList:
    if counter == 201:
        break
    if word not in finalListKeywords:
        finalListKeywords.append(word)
        counter = counter + 1
    else:
        continue

# Write file
f = open('/Users/yanisa/GoogleDrive/Publications_Conferences/Code/2020.CovidMetaphorMetonymyBookChptCollab/DataAnalysis/ListsOfKeywords/{}_200keywords.txt'.format(newsSource), 'w')
f.write('\n'.join(finalListKeywords))
