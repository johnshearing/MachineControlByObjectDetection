from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='Sample_Input_01.mp4')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()




## [Create Background Subtractor objects]
if args.algo == 'MOG2':
    # backSub = cv.createBackgroundSubtractorMOG2(history = 500, varThreshold = 250, detectShadows = True) # Original values
    backSub = cv.createBackgroundSubtractorMOG2(history = 5000, varThreshold = 1000, detectShadows = True) # Seems to work best  
else:
    backSub = cv.createBackgroundSubtractorKNN()
## [End of: Create Background Subtractor objects]




## [capture]
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
## [End of: capture]




## [Obtain frame information using get() method]
frame_width = int(capture.get(3))

frame_height = int(capture.get(4))

frame_size = (frame_width,frame_height)

fps = int(capture.get(cv.CAP_PROP_FPS))
## [End of: Obtain frame information using get() method]




## [Initialize video writer object]
output = None
output = cv.VideoWriter('./output_video_from_file.avi', cv.VideoWriter_fourcc(*'XVID'), fps, frame_size)



# Used to fade in the image.
img1_weight = 0



## [Create a blank image. Used for fading in the first few frames. Gets rid of white artifacts caused by background subtraction]
# dummy_frame = np.full((frame_height,frame_width,3), (255, 255, 255), np.uint8) # All white frame
dummy_frame = np.full((frame_height,frame_width,3), (0, 0, 0), np.uint8) # All black frame  
## [End of: Create a blank image. Used for fading in the first few frames. Gets rid of white artifacts caused by background subtraction]



while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    
    # Increment the image weight for fade in from black. One percent for the first 100 frames.
    img1_weight += 0.01
    
    # if img1_weight goes up, then img2_weight goes down accordingly and vice versa.
    img2_weight = 1 - img1_weight      
    
    
    ## [Fade in the first 100 frames from black. Gets rid of white artifacts caused by background subtraction]
    if capture.get(cv.CAP_PROP_POS_FRAMES) < 100 :
        
        # Fade in from black
        dst = cv.addWeighted(frame, img1_weight , dummy_frame, img2_weight , 0)   
        
        # Make the replacement
        frame = dst
    ## [End of: Fade in the first 100 frames from black. Gets rid of white artifacts caused by background subtraction]
    
  
    
    
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
            
                # Draw a green contour around the detected object or object cluster
                cv.drawContours(frame, contour, -1, (0,255,0), 3)
                
                
                
	            # create hull array for convex hull points
                hull = []
                # creating convex hull object for each contour
                hull.append(cv.convexHull(contour, False))
                # draw convex hull object in red
                # cv.drawContours(frame, hull, -1, (0, 0, 255), 3) # Drawing this further down in the code. No need to do it twice.

                # Find the area of the hull.
                hull_area = cv.contourArea(hull[0])
                area_ratio = area / hull_area

                ## [Find the moment of the contour]
                M = cv.moments(contour)

                # Use the moment to calculate the centroid
                if not (M['m00'] == 0):
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # Draw a circle at the centroid to indicate the contour
                    #cv.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
                    
                
                # Define an area-ratio threshold that indicates a cluster candidate.
                candidate_cluster = .87
                

                # Create a new hull.
                new_hull = cv.convexHull(contour,returnPoints = False) 
                
                # find the convexity defects in the new_hull
                defects = cv.convexityDefects(contour,new_hull)
                
                
                # Find the greatest convexity defect on the contour.  
                maxInColumns = np.amax(defects, axis=0)
                # print('Max value of every column: ', maxInColumns)
                maxOf_d = maxInColumns[0][3]
                # print('Max value of d is : ', maxOf_d)                 
                
                
                # Draw a red line to create the hull and a red dot to mark each defect.
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])
                    distance = d
                    
                    if distance == maxOf_d:
                        # draw a line segment on the hull corresponding to the greatest defect.
                        cv.line(frame,start,end,[0,0,255],2)
                        
                        # Draw a green dot over the defect.
                        cv.circle(frame,far,5,[0,0,255],-1)

                        
                        # draw a green dot over the mid point of the line segment on the hull with greatest defect.
                        x_m_point = (start[0] + end[0])//2
                        y_m_point = (start[1] + end[1])//2 
                        midpoint = tuple([x_m_point, y_m_point])
                        cv.circle(frame,midpoint,5,[0,255,0],-1)



                        # Take a color sample of the area around this point.
                        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                        
                        
                        
                        sample_width = 20
                        sample_height = 20                        
                        sample_x_origin = x_m_point - sample_width // 2
                        sample_y_origin = y_m_point - sample_height // 2  
                        cv.rectangle(frame, (sample_x_origin, sample_y_origin), (sample_x_origin + sample_width, sample_y_origin + sample_height), (0, 255, 0), 2)   
                        avg_hsv_color = np.array(cv.mean(hsv_frame[sample_y_origin : sample_y_origin + sample_height, sample_x_origin : sample_x_origin + sample_width])).astype(np.uint8)
                        print('Average color (hsv): ', avg_hsv_color)                        


                        # Assemble an HSV color value message.
                        col_val_msg = "The color is " + str(avg_hsv_color)
                        
                        # Display the HSV color.
                        cv.putText(frame, col_val_msg, (cx, cy - 30), cv.FONT_HERSHEY_SIMPLEX, 0.7 , (0,0,255)) 


                        # Display color range.
                        cv.putText(frame, "Color Range (41 to 73) (93 to 116) (112 to 171)", (cx, cy - 60), cv.FONT_HERSHEY_SIMPLEX, 0.7 , (0,0,255)) 
                        

    
                        # Check if the color sample is black
                        if (avg_hsv_color[0] >= 41) and  (avg_hsv_color[0] <= 73) and (avg_hsv_color[1] >= 93) and  (avg_hsv_color[1] <= 116) and (avg_hsv_color[2] >= 112) and  (avg_hsv_color[2] <= 171):
                            in_range = True
                            range_msg = "Hull midpoint is inside the black color range"
                        else:
                            range_msg = "Hull midpoint is outside the black color range"
                            in_range = False
                
                        # Display the color range message
                        cv.putText(frame, range_msg, (cx, cy - 90), cv.FONT_HERSHEY_SIMPLEX, 0.7 , (0,0,255)) 
                        
                        
                        
                        
                        # Put the smallest possible rectangle around the contour.
                        smallest_rect = cv.minAreaRect(contour)
                        box = cv.boxPoints(smallest_rect)
                        box = np.int0(box)
                        cv.drawContours(frame,[box],0,(0,0,255),2)
                        
                        
                        # Get the area of the smallest rectangle.
                        area_smallest_rect = cv.contourArea(box)
                        rectangle_ratio = hull_area / area_smallest_rect
                        rect_ratio_msg = "The hull to smallest rectangle ratio is " + str(rectangle_ratio)
                        # Display rect_ratio_msg
                        cv.putText(frame, rect_ratio_msg, (cx, cy - 120), cv.FONT_HERSHEY_SIMPLEX, .7 , (0,0,255))           
                        
                        
                        # Display the area at the centroid of each detected package or cluster of packages
                        cv.putText(frame, "The contour to hull ratio is " + str(area_ratio), (cx, cy - 150), cv.FONT_HERSHEY_SIMPLEX, 0.7 , (0,0,255))                         
                        
                        
                        
                        # Determine if we have a cluster
                        if (in_range == True) and  (area_ratio < .8) and (rectangle_ratio < .8):
                            cluster_msg = "Cluster"
                        else:
                            cluster_msg = ""              
                            
                        # Display the word cluster
                        cv.putText(frame, cluster_msg, (cx, cy + 40), cv.FONT_HERSHEY_COMPLEX, 2.1 , (0,0,255))                        
                        
                        
                        

                    
    ## [End of: Filter out contours with small areas and overlay the big contours onto the original images]
    
    


    ## [show]
    if capture.get(cv.CAP_PROP_POS_FRAMES) > 2 :
        # cv.imshow('blank', blank_image)
        cv.imshow('FG Mask', fgMask)        
        cv.imshow('Frame', frame)                 
    ## [End of: show]
    
    

    
    ## [write the frame to the output files]
    # result = np.asarray(frame)
    # result = cv.cvtColor(result, cv.COLOR_RGB2BGR) 
    # output.write(frame)
    output.write(frame)
    ## [End of: write the frame to the output files]
    
    


    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    
    
 

# Release the objects
capture.release()
output.release()
