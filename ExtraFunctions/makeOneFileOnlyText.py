import os

date = 'May22'
newsSource = 'Fox'

nameOfFirstFilePath = '/Users/yanisa/GoogleDrive/Publications_Conferences/Code/2020.CovidMetaphorMetonymyBookChptCollab/FinalData/{}/FinalData/'.format(newsSource)
firstFilePath = os.listdir(nameOfFirstFilePath)

# Open the file, take only lines 1 & 7 (headline + text) and write it to a new file
firstNewSourceLongFileWrite = open('/Users/yanisa/GoogleDrive/Publications_Conferences/Code/2020.CovidMetaphorMetonymyBookChptCollab/FinalData/{}/AllText{}.txt'.format(newsSource, date), 'w')
for directory in firstFilePath:
    if os.path.isdir(nameOfFirstFilePath + directory) == True:
        files = os.listdir(nameOfFirstFilePath + directory)
        for eachFile in files:
            if eachFile == '.DS_Store':
                continue
            else:
                readEachFile = open(nameOfFirstFilePath + directory + '/' + eachFile).read()
                fixedFile = readEachFile.split('\n') 
                print(eachFile)
                firstNewSourceLongFileWrite.write(fixedFile[0] + '\n')
                firstNewSourceLongFileWrite.write(fixedFile[6] + '\n')