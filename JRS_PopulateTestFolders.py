'''
Moves every 12th image and annotation file into a test folder for training the AI.
After running the test folder will have 20% of the source folder
PATH is the source folder.
DEST is the destination folder.
The destination folder must exist or the program will not run.
'''

import os

PATH = 'C:/Users/jshea/Desktop/ML/MachineControlByObjectDetection/bad20220131_1300/'


DEST = 'C:/Users/jshea/Desktop/ML/MachineControlByObjectDetection/bad20220131_1300/test/'



filenames = os.listdir(PATH)

for i in range(0, len(filenames), 12):
    os.rename(PATH + filenames[i], DEST+ filenames[i])    
    
    
for i in range(1, len(filenames), 12):  
    os.rename(PATH + filenames[i], DEST + filenames[i])     
