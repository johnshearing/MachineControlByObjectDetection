'''
Removes jpg files that do not have a matching txt file
Removes txt files that do not have a matching jpg file.
A dialog box will prompt for the source folder which contains the files to be checked for orphan files.
'''

import os
import sys
from tkinter import filedialog
from tkinter import *

from absl import app, flags, logging
from absl.flags import FLAGS


flags.DEFINE_boolean('delete_mode', False, '\nBy default the program only reports which orphan files should be deleted.\nIf delete_mode flag is set then program will actually delete orphans files.')



def main(_argv):

    root = Tk()
    root.withdraw()
    
    
    PATH = filedialog.askdirectory(title = "Select Target Directory") 
    
    try:
        filenames = os.listdir(PATH)
    except:
        print("No target directory was selected.")
        sys.exit()
        
    
    
    
    for i in range(0, len(filenames)):
        
        # if it ends in .txt
        if filenames[i].endswith(".txt"):
    
            if filenames.count(filenames[i].replace(".txt", ".jpg")) != 1:
                print(filenames[i])
                
                if FLAGS.delete_mode == True:
                    os.remove(PATH + "/" + filenames[i])
                
                
                
        # if it ends in .txt
        if filenames[i].endswith(".jpg"):
    
            if filenames.count(filenames[i].replace(".jpg", ".txt")) != 1:
                print(filenames[i]) 
                
                if FLAGS.delete_mode == True:
                    os.remove(PATH + "/" + filenames[i])  
    
 
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass     