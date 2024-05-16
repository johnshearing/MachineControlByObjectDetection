'''
Adjust camera properties for best results using AMCap.exe before running detections with webcam.
AMCap.exe can be downloaded at https://www.arducam.com/downloads/app/AMCap.exe

The camera we are using can be purchased here:
https://www.amazon.com/Varifocal-3840x2160-Adjustable-Conference-Streaming/dp/B08BHXF9R1/ref=sr_1_1?keywords=B08BHXF9R1&link_code=qs&qid=1643295841&sourceid=Mozilla-search&sr=8-1&th=1

The USB relay we are using can be purchased here:
https://www.amazon.com/dp/B01CN7E0RQ?psc=1&ref=ppx_yo2_dt_b_product_details

This program is used with or without a USB relay to collect .jpg images and annotation .txt files used to train the AI.
A hull contour is drawn in red around each package or cluster in the images.
Also the smallest rectangle rotated to fit around each of the packages or cluster of packages are drawn in green.
The color or the output images does not appear correct to the human eye but these are the colors needed to train the AI. 
So don't mess with the colors of the output images.'
We will feed these augmented images to the AI for training and augmented images will also be fed to the AI when it is being used to find clusters.
The augmented the images should help the AI better select what is a cluster and what is not.


When creating augmented images of clusters, the program also creates an annotation .txt file which specifies the class index and class name (0 for Bad in this case) 
and also specifies the coordinates of the smallest bounding rectangle drawn in the x-y coordinate plane which will fit around the packages or clusters of packages 
and the smallest rotated rectanges which are drawn around these.
These anotation files along with the augmented images are used for training the AI.


With relay present in USB port, press the letter "b" on the keyboard to reject clustered packages at the VarioRoute 1600 and collect images of the clusters in the ./Bad folder for training the AI.
If no relay is present in a USB port, press the letter "b" on the keyboard to collect images of clusters and annotation files in the ./Bad folder for training the AI.
Pressing the letter "g" allows packages to pass by the VarioRoute but collects images of well singulated packages and annotation files in the ./Good folder for training the AI.

Before using this program, install Anaconda and then all the imported packages at the Anaconda command prompt.
The following are the commands to install the required packages at the Anaconda prompt.
conda install -c anaconda opencv
conda install -c anaconda pyserial

When you need to stop the program, press the letter "q" or the esc key

To use this program: 
1. Open the Anaconda command line window.
2. cd into the folder where this program resides.
3. Then enter any of the following commands to start the program.

The example command below runs the program using the webcam at index 0 for input and supresses video output.
python CollectTrainingImages.py --input "0" --output ""

The example command below runs the program using input video specifed in double quotes and produces output video also specifed in double quotes.
python CollectTrainingImages.py --input "./my_input_video.mp4" --output "my_output_video.avi"

The command below runs the program with hard coded defaults.
Input video will be from Sample_Input_01.mp4
Output video will go to output_file.avi
python CollectTrainingImages.py
'''



import cv2 as cv
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import datetime
import time
import serial
import copy


try:
    using_relay = True
    ser=serial.Serial(port='com3',baudrate=9600)
    ser.close()
    ser.open()
except:
    using_relay = False



parser = argparse.ArgumentParser(description='Collect training images.\nUsed to train neural network.\nPress letter b for bad (clustered) or press letter g for good (well singulated).', formatter_class=RawTextHelpFormatter)
                                              
# Specify an index (usually 0 or 1) to indicate webcam input.
# Specify a path and file name to indicate video input.
parser.add_argument('--input', type=str, help='Path to an input video file (mp4 or avi) or a webcam index (usually 0 or 1).\nThe default input is ./data/video/Sample_Input_01.avi\n\n', default='./data/video/Sample_Input_01.avi')

# Specify a path and file name to indicate output video file. avi extensions only are acceptable.
# Specify an empty string in default below to suppress video output.
parser.add_argument('--output', type=str, help='Path and name of the saved .avi file after processing.\n--output "" to suppress video output.\nDefault is ./detections/collector_output_video.avi\n\n', default='./detections/collector_output_video.avi')

parser.add_argument('--good_folder', type=str, help='Path to folder where classified good images are saved.\nPress the letter g on the keyboard to save frame in Good folder.\nA blank annotation txt file of the same name is also created.\nThe default is ./good\n\n', default='./good')
parser.add_argument('--bad_folder', type=str, help='Path to folder where classified bad images (images of clusters) are saved.\nPress the letter b to save frame in Bad folder.\nAn annotation txt file of the same name is also created.\nThe annotation file contains the class indexes and the coordinates of the bounding rectangles in the x-y plain.\nThe default is ./bad\n\n', default='./bad')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN or MOG2).\nThe default is MOG2', default='MOG2')
args = parser.parse_args()


bboxes = []
relay_was_triggered = False


## [Create Background Subtractor objects]
if args.algo == 'MOG2':
    # backSub = cv.createBackgroundSubtractorMOG2(history = 500, varThreshold = 250, detectShadows = True) # Original values
    backSub = cv.createBackgroundSubtractorMOG2(history = 5000, varThreshold = 1000, detectShadows = False) # These values seem to work best.   
else:
    backSub = cv.createBackgroundSubtractorKNN()
## [End of: Create Background Subtractor objects]



## [capture]
try: # Will only succeed if input is given as a number which indicates that input is a camera.
    capture = cv.VideoCapture(int(int(args.input) + cv.CAP_DSHOW))
    source = "camera"
    # JRS used for learning how to change camera settings. This is not working. May have to use Arducam's library and API
    # capture.set( cv.CAP_PROP_BRIGHTNESS, -64 )
    # capture.set( cv.CAP_PROP_CONTRAST, 0 )
    # capture.set( cv.CAP_PROP_HUE, -1 )
    # capture.set( cv.CAP_PROP_SATURATION, 62 )
    # capture.set( cv.CAP_PROP_SHARPNESS, 6 )
    # capture.set( cv.CAP_PROP_GAMMA, 100 )
    # capture.set( cv.CAP_PROP_GAIN, 100 )
    # capture.set( cv.CAP_PROP_EXPOSURE, -4 )
    # capture.set( cv.CAP_PROP_FPS, 10 )
    
    # capture.set( cv.CAP_PROP_FRAME_HEIGHT, 480 )
    # capture.set( cv.CAP_PROP_FRAME_WIDTH, 640 )    
    # capture.set( cv.CAP_PROP_FRAME_HEIGHT, 1280 ) 
    # capture.set( cv.CAP_PROP_FRAME_WIDTH, 720 )
    
    # print("CAP_PROP_BRIGHTNESS is {}  :".format(capture.get(cv.CAP_PROP_BRIGHTNESS)))
    # print("CAP_PROP_CONTRAST is {}  :".format(capture.get(cv.CAP_PROP_CONTRAST)))
    # print("CAP_PROP_HUE is {}  :".format(capture.get(cv.CAP_PROP_HUE)))
    # print("CAP_PROP_SATURATION is {}  :".format(capture.get(cv.CAP_PROP_SATURATION)))
    # print("CAP_PROP_SHARPNESS is {}  :".format(capture.get(cv.CAP_PROP_SHARPNESS)))
    # print("CAP_PROP_GAMMA is {}  :".format(capture.get(cv.CAP_PROP_GAMMA)))
    # print("CAP_PROP_GAIN is {}  :".format(capture.get(cv.CAP_PROP_GAIN)))
    # print("CAP_PROP_HEIGHT is {}  :".format(capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
    # print("CAP_PROP_WIDTH is {}  :".format(capture.get(cv.CAP_PROP_FRAME_WIDTH)))
    # print("CAP_PROP_FPS : '{}'".format(capture.get(cv.CAP_PROP_FPS)))
    # print("CAP_PROP_EXPOSURE : '{}'".format(capture.get(cv.CAP_PROP_EXPOSURE)))
except: # Input is not a number so input must be from a video source.
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
    source = "video"    


if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
## [End of: capture]




## [Get and set information required to write an .avi file and initialize video writer object]
frame_width = int(capture.get(3))

frame_height = int(capture.get(4))

frame_size = (frame_width,frame_height)

if source == "camera":
    fps = 30 # We explicitly define the frame rate if the input is camera otherwise fps will be zero and no .avi recording will be made.
else: # Source is video input. We can read the frame rate from that
    fps = int(capture.get(cv.CAP_PROP_FPS))
    
output = None

if args.output != "":
    output = cv.VideoWriter(args.output, cv.VideoWriter_fourcc(*'XVID'), fps, frame_size)
    
## [End of: Get and set information required to write an .avi file and initialize video writer object]




# Used to fade in the image. Gets rid of white artifacts caused by background subtraction
img1_weight = 0
frame_count = 0


## [Create a blank image (all black). Used for fading in the first few frames. Gets rid of white artifacts caused by background subtraction]
# dummy_frame = np.full((frame_height,frame_width,3), (255, 255, 255), np.uint8) # All white frame
dummy_frame = np.full((frame_height,frame_width,3), (0, 0, 0), np.uint8) # All black frame  
## [End of: Create a blank image (all black). Used for fading in the first few frames. Gets rid of white artifacts caused by background subtraction]





while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    
    # Increment the image weight for fade in from black. One percent for the first 100 frames.
    img1_weight += 0.01
    
    # if img1_weight goes up, then img2_weight goes down accordingly and vice versa.
    img2_weight = 1 - img1_weight      
    
    
    ## [Fade in the first 100 frames from black. Gets rid of white artifacts caused by background subtraction]
    if frame_count < 100 :
        
        # Fade in from black
        dst = cv.addWeighted(frame, img1_weight , dummy_frame, img2_weight , 0)   
        
        # Make the replacement
        frame = dst
    ## [End of: Fade in the first 100 frames from black. Gets rid of white artifacts caused by background subtraction]
    
  
    # inputVideo = copy.deepcopy(frame)
    inputVideo = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    
    ## [gaussian blur helps to remove noise These settings seem to remove reflections from rollers.] 
    blur = cv.GaussianBlur(frame, (0,0), 5, 0, 0)
    ## [End of: gaussian blur helps to remove noise These settings seem to remove reflections from rollers.]
    



    ## [Remove the background and produce a grays scale image]
    fgMask = backSub.apply(blur)
    ## [End of: Remove the background and produce a grey scale image]
    


    
    ## [Apply morphological operations to ensure we have a good mask.]
    kernel = None
    fgMask = cv.dilate(fgMask, kernel, iterations = 2) # This function seems to help
    ## [End of: Apply morphological operations to ensure we have a good mask.]
    
    
    
    
    ##[Get rid of gray artifacts. Produces a black and white image]
    (thresh, fgMask) = cv.threshold(fgMask, 127, 255, cv.THRESH_BINARY)    
    ##[End of: Get rid of gray artifacts. Produces a black and white image]
    
    

    
    ##[Get rid of black artifacts inside white BLOBs representing packages. Turns black pixes to white when next to white]
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(100,100)))    
    ##[End of: Get rid of black artifacts inside white BLOBs representing packages. Turns black pixes to white when next to white]   

    
    
    
    ## [Finding contours without canny]
    contours, _ = cv.findContours(fgMask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)    
    ## [End of: Finding contours without canny]
    

    

    ## [Filter out contours with small areas and overlay the big contours onto the original images then calculate convexity]
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 5000:
                
            # Draw a green contour around the detected object or object cluster.
            # This is just to help us humans know what the camera sees as a package.
            # or to put it another way, how well the Vision System is able to differentiate the package from the background. 
            # cv.drawContours(frame, contour, -1, (0,255,0), 3)
                
            #create hull array for convex hull points
            hull = []
            # creating convex hull object for each contour
            hull.append(cv.convexHull(contour, False))
            # draw convex hull object in blue
            cv.drawContours(frame, hull, -1, (255, 0, 0), 3)

                        
            # Put the smallest possible rectangle around the contour by rotating the rectangle for best fit.
            smallest_rect = cv.minAreaRect(contour)
            box = cv.boxPoints(smallest_rect)
            box = np.int0(box)
            # Draw smallest rectangle in green
            cv.drawContours(frame,[box],0,(0,255,0),2)
                
                
            # Collect information defining a bounding rectangle for each contour in the xy coordinate system.
            contours_poly = cv.approxPolyDP(box, 3, True)
            boundRect = cv.boundingRect(contours_poly)

            # Note the coordinates of the bounding rectangle and convert to ratios. Will be used for making a classifier text file.
            # Find the center of the bounding rectangle.
            midpoint = ((int(boundRect[0]) + int(boundRect[2]) // 2), ((int(boundRect[1]) + int(boundRect[3]) // 2)))
            
            # Divide center x of bounding box by total frame width.
            midpoint_x_ratio = (int(boundRect[0]) + int(boundRect[2]) // 2) / frame_width
            
            # Divide center y of bounding box by total frame height.
            midpoint_y_ratio = (int(boundRect[1]) + int(boundRect[3]) // 2) / frame_height
            
            # Divide width of bounding box by total frame width.
            width_ratio = int(boundRect[2]) / frame_width
            
            # Divide height of bounding box by total frame height.
            height_ratio = int(boundRect[3]) / frame_height
            
            bboxes.append(str(midpoint_x_ratio) + " " + str(midpoint_y_ratio) + " " + str(width_ratio) + " " + str(height_ratio))
            
            '''
            # The following code is for testing that the bounding rectange is positioned correctly.
            # We are not actually drawing the bounding rectangle in the xy coordinate system nor marking its center.
            # Rather we export bounding box information from above to a text file along with a jpg image for use in training the object classifier 
            
            # Draw a circle at the center of the bounding rectangle just for testing.
            cv.circle(frame, midpoint, 5, [0,0,255], -1)            
            
            # Draw bounding rectangle in red.               
            # cv.rectangle(frame, (int(boundRect[0]), int(boundRect[1])), \
            # (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (0,0,255), 2)
            '''

    ## [End of: Filter out contours with small areas and overlay the big contours onto the original images]            

    
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)    


    keyboard = cv.waitKey(1)
    if keyboard == ord('q') or keyboard == 27:
        break

    elif keyboard == ord('b'):

        if bboxes: # if prevents error when there is nothing to classify.
            #Save the frame as an image in the bad folder.    
            t = datetime.datetime.now()
            ts = t.strftime("%Y") + t.strftime("%m") + t.strftime("%d") + t.strftime("%H") + t.strftime("%M") + t.strftime("%S") + t.strftime("%f")
            # Write the image to disk
            cv.imwrite(args.bad_folder + "/" + ts + ".jpg", frame)
            
    
            for i in bboxes:
                # Write a text file with the class index and the coordinates of the bounding rectangles. 
                # Save this file in the Bad folder with the same root name as the image but with the .txt extension.
                # Used to train the object classifier. 
                # Will be class 0 Which represents Bad in the classes.txt file and represents clustered packages.
                f = open(args.bad_folder + "/" + ts + ".txt", "a")
                f.write("0 " + i + "\n")
                    
            f.close()      

        if using_relay == True:
            relay_was_triggered = True
            ser.write(bytes.fromhex("A0 01 01 A2"))
            # print('Circuit closed. Button is pressed')    
            time.sleep(.4) # Keeps the relay closed on long press. Otherwise debouce in Windows keyboard releases the relay uncommanded.
        
    elif keyboard == ord('g'):
        
        if using_relay == True:
            relay_was_triggered = False
            ser.write(bytes.fromhex("A0 01 00 A1"))
            # print('Circuit Open. Button is released')
        
        

        #Save the frame as an image in the good folder.    
        t = datetime.datetime.now()
        ts = t.strftime("%Y") + t.strftime("%m") + t.strftime("%d") + t.strftime("%H") + t.strftime("%M") + t.strftime("%S") + t.strftime("%f")
        # Write the image to disk
        cv.imwrite(args.good_folder + "/" + ts + ".jpg", frame)
        

        # Write an empty text file of the same root name as the image but with .txt extension. 
        # No classes or bounding rectangle coordinates are specifed because we are telling the classifier that there is nothing to classify in this image. 
        # In other words, everything is well singulated.
        f = open(args.good_folder + "/" + ts + ".txt", "a")
        f.write("")
        f.close()
            
    elif keyboard == -1:
        if using_relay == True:
            ser.write(bytes.fromhex("A0 01 00 A1"))
            # print('Circuit Open. Button is released')
            relay_was_triggered = False     
            





    ## [show]
    cv.imshow('Frame', frame)    
    # cv.imshow('FG Mask', fgMask)        
    ## [End of: show] 
    

    ## [write the frame to the output file]
    if args.output != "":
        # output.write(frame)
        output.write(inputVideo)        
    ## [End of: write the frame to the output file]
        
    
    bboxes = [] # Empty the array for the next frame.
    frame_count = frame_count + 1
 

# Release the objects
capture.release()

if args.output != "":
    output.release()
