import os

date = 'July9'
newsSource = 'Conservative'

nameOfFirstFilePath = '/Users/yanisa/GoogleDrive/Data/MyResearchData/2020_LibConservMediaData/Articles/{}Articles_All/'.format(newsSource)
firstFilePath = os.listdir(nameOfFirstFilePath)

# Open the file, take only lines 1 & 7 (headline + text) and write it to a new file
firstNewSourceLongFileWrite = open('/Users/yanisa/GoogleDrive/Data/MyResearchData/2020_LibConservMediaData/Articles/TextOnly_OneFile_LibAndConserv/AllText{}.txt'.format(newsSource, date), 'w')
for eachFile in firstFilePath:
    if eachFile == '.DS_Store':
        continue
    else:
        readEachFile = open(nameOfFirstFilePath + eachFile).read()
        fixedFile = readEachFile.split('\n') 
        print(eachFile)
        firstNewSourceLongFileWrite.write(fixedFile[0] + '\n')
        firstNewSourceLongFileWrite.write(fixedFile[6] + '\n')