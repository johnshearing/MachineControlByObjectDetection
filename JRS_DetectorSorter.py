import os

'''
ToDo
Play with the size of the circle used to span undetected areas. 
Consider adding more lighting for sharper images and for better recognition of darker objects.
Comment all the code even more.
Update the wiki page.
Make a GUI.
Get more images of packages side by side then retrain the AI.
Consider switching to a solid state relay for less power comsumption and greater reliability.
'''


'''
Adjust camera properties for best results using AMCap.exe before running detections with webcam.
AMCap.exe can be downloaded at https://www.arducam.com/downloads/app/AMCap.exe

The camera we are using can be purchased here.
https://www.amazon.com/Varifocal-3840x2160-Adjustable-Conference-Streaming/dp/B08BHXF9R1/ref=sr_1_1?keywords=B08BHXF9R1&link_code=qs&qid=1643295841&sourceid=Mozilla-search&sr=8-1&th=1

The USB relay we are using can be purchased here:
https://www.amazon.com/dp/B01CN7E0RQ?psc=1&ref=ppx_yo2_dt_b_product_details
'''

# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Start JRS Mod
import datetime
import serial
# End JRS Mod

import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import copy


try:
    using_relay = True
    ser=serial.Serial(port='com3',baudrate=9600)
    ser.close()
    ser.open()
except:
    using_relay = False
    
    
    
    


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/custom-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/Sample_Input_01.avi', 'Path to input video or set to 0 or 1 for webcam index')
flags.DEFINE_string('output_video', "./detections/video/detector_results.avi", 'Path to output video.\nUse "none" to suppress video output')
flags.DEFINE_string('output_format', 'XVID', 'Codec used in VideoWriter when saving video to file')
flags.DEFINE_string('output_images', './detections/images', 'Path to folder where images of detections are stored.\nUse "none" to suppress image output')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.2, 'Score threshold. Controls sensitivity of the cluster detector.\nRanges between .01 and 1 are valid.\nLower numbers make it more sensitive but increase the possibility of false positives.\nIf you have a well trained model then you can use a low value without getting false positives.\nPlay with this value.') 
flags.DEFINE_boolean('count', True, 'Show count of objects on screen. Prints class and count of objects detected to console. Also prints if USB relay is open or closed') 
flags.DEFINE_boolean('dont_show', False, 'Dont show video output')
flags.DEFINE_boolean('info', False, 'Print info on detections - Class and bounding box coordinates')
flags.DEFINE_boolean('crop', False, 'Crop detections from images')
flags.DEFINE_boolean('plate', False, 'Perform license plate recognition')
flags.DEFINE_string('algo', 'MOG2', 'Background subtraction method (KNN, MOG2).')


flags.DEFINE_float('check_zone', 0.35, 'Ratio of full screen (starting from right side) to be checked for clusters.\nValues between .1 and .9 are valid\nWe need to keep the check zone as small as possible.\nThis so relay will activate near photoeye as packages enter frame.\nThis is how system knows what package to reject.\nBut the zone must be large enough so that the clusters can be detected.\nA value of .4 seems to be a good balance.')
flags.DEFINE_integer('too_wide', 180, 'Measured in pixels from the top of frame.\nVarioRoute 1600 will reject packages less than this value\nGreen dot on screen gives a visual indication of what will be rejected.')
flags.DEFINE_integer('reset_rate', 5, 'Will reset p_b_box_pos to zero every so many frames as defined by this variable.\nThis prevents the oversize and clustered package tests from locking up if the AI loses track of the bounding box.\nThis parameter works together with the check_zone parameter.\nIf one is changed then the other will need to be changed also.\nreset_rate should nearly equal the number of frames required to move a package from the right side of the frame past the check_zone\nIf set to high then following clusters will not be rejected.\nSo we are trying to find a balance\nWe want to avoid activating the relay twice on the same cluster and we want to catch any following clusters for rejection.')
flags.DEFINE_string('good_folder', './good', 'Path to folder where classified good images are saved.\nPress the letter g on the keyboard to save frame in Good folder.\nA blank annotation txt file of the same name is also created.')
flags.DEFINE_string('bad_folder', './bad', 'Path to folder where classified bad images (clusters) are saved.\nPress the letter b to save frame in Bad folder.\nAn annotation txt file of the same name is also created.\nThe annotation file contains the class indexes and the coordinates of the bounding rectangles in the x-y plane.')







def main(_argv):
    

    # Suppress over size package measurement when first starting AI
    # The first one is always a false positive so we don't want the relay to activate.
    starting = True
    
    
    # Stop the relay from activating on an oversize package or cluster of packages more than once.
    # If bounding box position decreases with respect to the previous bounding box postion then we are looking at the same bounding box as before in the previous video frame. 
    # So don't activate the relay again.
    # If bounding box position increases with respect to the previous bounding box postion then we are looking at a new bounding box in this new video frame. 
    # So go ahead and activate the relay to reject package(s)
    # Keeping track of the upper left corner of the bounding box.
    p_b_box_pos = 0 # previous_bounding_box_position
    
    # Used with p_b_box_pos above.
    # Will reset p_b_box_pos to zero every 8 frames.
    # This prevents the oversize package test from locking up if it loses track of the bounding box.
    f_cnt_s_o_size = 0 # frame_count_since_over_size_package
    
    
    # Used with p_b_box_pos above.
    # Will reset p_b_box_pos to zero every 8 frames.
    # This prevents the cluster test from locking up if it loses track of the bounding box.
    f_cnt_s_cluster = 0 # frame_count_since_cluster
    
    
    # Used to create annotation text files which hold class index and bounding box coordinates in the x-y plane.
    # These text files along with saved images will be used to train the AI.
    anot_bboxes = [] # annotation_bounding_boxes
    
    
    # Used to stop relay from rejecting packages.
    suppress_relay = False
    
    
    
    ## [Create Background Subtractor objects]
    if FLAGS.algo == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2(history = 5000, varThreshold = 1000, detectShadows = False) # These values seem to work best.   
    else:
        backSub = cv2.createBackgroundSubtractorKNN()
    ## [End of: Create Background Subtractor objects]    
    

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    # get video name by using split method
    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[0]
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
        
        
        
        
    # Select which camera.
    camera_number = FLAGS.video    

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(int(camera_number) + cv2.CAP_DSHOW))
        
        # vid.set( cv2.CAP_PROP_BRIGHTNESS, -64 )
        # vid.set( cv2.CAP_PROP_CONTRAST, 0 )
        # vid.set( cv2.CAP_PROP_HUE, -1 )
        # vid.set( cv2.CAP_PROP_SATURATION, 62 )
        # vid.set( cv2.CAP_PROP_SHARPNESS, 6 )
        # vid.set( cv2.CAP_PROP_GAMMA, 100 )
        # vid.set( cv2.CAP_PROP_GAIN, 100 )
        # vid.set( cv2.CAP_PROP_EXPOSURE, -4 )
        # vid.set( cv2.CAP_PROP_FPS, 10 )        
        
        vid.set( cv2.CAP_PROP_FRAME_HEIGHT, 480 )
        vid.set( cv2.CAP_PROP_FRAME_WIDTH, 640 )
        print("CAP_PROP_BRIGHTNESS is {}  :".format(vid.get(cv2.CAP_PROP_BRIGHTNESS)))
        print("CAP_PROP_CONTRAST is {}  :".format(vid.get(cv2.CAP_PROP_CONTRAST)))
        print("CAP_PROP_HUE is {}  :".format(vid.get(cv2.CAP_PROP_HUE)))
        print("CAP_PROP_SATURATION is {}  :".format(vid.get(cv2.CAP_PROP_SATURATION)))
        print("CAP_PROP_SHARPNESS is {}  :".format(vid.get(cv2.CAP_PROP_SHARPNESS)))
        print("CAP_PROP_GAMMA is {}  :".format(vid.get(cv2.CAP_PROP_GAMMA)))
        print("CAP_PROP_GAIN is {}  :".format(vid.get(cv2.CAP_PROP_GAIN)))
        print("CAP_PROP_HEIGHT is {}  :".format(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("CAP_PROP_WIDTH is {}  :".format(vid.get(cv2.CAP_PROP_FRAME_WIDTH)))
        print("CAP_PROP_FPS : '{}'".format(vid.get(cv2.CAP_PROP_FPS)))
        print("CAP_PROP_EXPOSURE : '{}'".format(vid.get(cv2.CAP_PROP_EXPOSURE)))       
        source = "camera"        
    except:
        vid = cv2.VideoCapture(video_path)
        source = "video"
        vid.set( cv2.CAP_PROP_FRAME_HEIGHT, 480 )
        vid.set( cv2.CAP_PROP_FRAME_WIDTH, 640 )        


    out = None



    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if source == "video":
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        
    else: # Source is camera
        fps = 10        
    

    codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    
    if FLAGS.output_video != "none":    
        out = cv2.VideoWriter(FLAGS.output_video, codec, fps, (width, height))
        

        
        
        
        
        
        
    # Used to fade in the image. Gets rid of white artifacts caused by background subtraction
    img1_weight = 0
    frame_num = 0
    
    ## [Create a blank image (all black). Used for fading in the first few frames. Gets rid of white artifacts caused by background subtraction]
    dummy_frame = np.full((height, width, 3), (0, 0, 0), np.uint8) # All black frame  
    ## [End of: Create a blank image (all black). Used for fading in the first few frames. Gets rid of white artifacts caused by background subtraction]      
    
        
        
        


    while True:
        return_value, frame = vid.read()
        
        if frame is None:
            print('Video has ended or failed, try a different video format!')
            break        
        
        
        # Increment the image weight for fade in from black. One percent for the first 100 frames.
        img1_weight += 0.01
        
        # if img1_weight goes up, then img2_weight goes down accordingly and vice versa.
        img2_weight = 1 - img1_weight      
        
        
        ## [Fade in the first 100 frames from black. Gets rid of white artifacts caused by background subtraction]
        if frame_num < 100 :
            
            # Fade in from black
            dst = cv2.addWeighted(frame, img1_weight , dummy_frame, img2_weight , 0)   
            
            # Make the replacement
            frame = dst
            
            
            ## [gaussian blur helps to remove noise These settings seem to remove reflections from rollers.] 
            blur = cv2.GaussianBlur(frame, (0,0), 5, 0, 0)
            ## [End of: gaussian blur helps to remove noise These settings seem to remove reflections from rollers.]
            
        
        
        
            ## [Remove the background and produce a grays scale image]
            fgMask = backSub.apply(blur)       
            ## [End of: Remove the background and produce a grey scale image]            
            
        else:
            ## [gaussian blur helps to remove noise These settings seem to remove reflections from rollers.] 
            blur = cv2.GaussianBlur(frame, (0,0), 5, 0, 0)
            ## [End of: gaussian blur helps to remove noise These settings seem to remove reflections from rollers.]
            
        
        
        
            ## [Remove the background and produce a grays scale image.]
            # Stop learning the background after 100 frames. 
            # This prevents OpenCV from thinking the packages are the background during times of high volume.
            fgMask = backSub.apply(blur, learningRate = 0)       
            ## [End of: Remove the background and produce a grey scale image]            
            
        ## [End of: Fade in the first 100 frames from black. Gets rid of white artifacts caused by background subtraction]        
        
        

            

        
    
    
        
        ## [Apply morphological operations to ensure we have a good mask.]
        kernel = None
        fgMask = cv2.dilate(fgMask, kernel, iterations = 2) # This function seems to help
        ## [End of: Apply morphological operations to ensure we have a good mask.]
        
        
        
        
        ##[Get rid of gray artifacts. Produces a black and white image]
        (thresh, fgMask) = cv2.threshold(fgMask, 127, 255, cv2.THRESH_BINARY)    
        ##[End of: Get rid of gray artifacts. Produces a black and white image]
        
        
    
        
        ##[Get rid of black artifacts inside white BLOBs representing packages. Turns black pixes to white when next to white]
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100)))    
        ##[End of: Get rid of black artifacts inside white BLOBs representing packages. Turns black pixes to white when next to white]   
    
        
        
        
        ## [Finding contours without canny]
        contours, _ = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)    
        ## [End of: Finding contours without canny]        
        
        
        raw_frame = copy.deepcopy(frame)
        
        
        ## [Filter out contours with small areas and overlay the big contours onto the original images then calculate convexity]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:
                    
                # Draw a green contour around the detected object or object cluster.
                # This is just to help us humans know what the camera sees as a package.
                # or to put it another way, how well the Vision System is able to differentiate the package from the background. 
                # cv2.drawContours(frame, contour, -1, (0,255,0), 3)
                    
                #create hull array for convex hull points
                hull = []
                # creating convex hull object for each contour
                hull.append(cv2.convexHull(contour, False))
                # draw convex hull object in blue
                cv2.drawContours(frame, hull, -1, (255, 0, 0), 3)
    
                            
                # Put the smallest possible rectangle around the contour by rotating the rectangle for best fit.
                smallest_rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(smallest_rect)
                box = np.int0(box)
                # Draw smallest rectangle in green
                cv2.drawContours(frame,[box],0,(0,255,0),2)        

                
                # Collect information defining a bounding rectangle for each contour in the xy coordinate system.
                contours_poly = cv2.approxPolyDP(box, 3, True)
                boundRect = cv2.boundingRect(contours_poly)
                
                
                # Note the coordinates of the bounding rectangle and convert to ratios. Will be used for making a classifier text file.
                # Find the center of the bounding rectangle.
                midpoint = ((int(boundRect[0]) + int(boundRect[2]) // 2), ((int(boundRect[1]) + int(boundRect[3]) // 2)))
                
                # Divide center x of bounding box by total frame width.
                midpoint_x_ratio = (int(boundRect[0]) + int(boundRect[2]) // 2) / width
                
                # Divide center y of bounding box by total frame height.
                midpoint_y_ratio = (int(boundRect[1]) + int(boundRect[3]) // 2) / height
                
                # Divide width of bounding box by total frame width.
                width_ratio = int(boundRect[2]) / width
                
                # Divide height of bounding box by total frame height.
                height_ratio = int(boundRect[3]) / height
                
                anot_bboxes.append(str(midpoint_x_ratio) + " " + str(midpoint_y_ratio) + " " + str(width_ratio) + " " + str(height_ratio))
                
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

                # Draw bounding rectangle in the x-y coordinate plane using color red. Just for testing oversize packages.             
                #cv2.rectangle(frame, (int(boundRect[0]), int(boundRect[1])), \
                #(int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (0,0,255), 2)
                
                
                # Prevents unwanted activation of relay when first starting the program
                if starting == False: 
                
                    # Check the position of the upper left corner of bounding box height and reject if oversized. 
                    # Prevents wide packages from exceeding the width of conveyor to cause a jam
                    if int(boundRect[1]) < FLAGS.too_wide:
                        
                        # Only check size on the right side of the frame. These are the packages coming into view.
                        if int(boundRect[0]) > frame_size[1] * (1 - FLAGS.check_zone):
                            print("")
                            print("Oversized Package")
                            
                            # Only activate relay on an oversize bounding box one time.
                            # In other words don't reject an oversize bounding box that has appeared in a previous frame.
                            if int(boundRect[0]) > p_b_box_pos:
                            
                            
                                if using_relay == True and suppress_relay == False:
                                    ser.write(bytes.fromhex("A0 01 01 A2")) # Activate relay
                                    
                                    # Needed to debounce the relay. PLC may not see that the relay has been triggered without this.
                                    time.sleep(.5) 
                                
                                
                            
                                if using_relay == True and suppress_relay == False:
                                    ser.write(bytes.fromhex("A0 01 00 A1")) # Deactivate relay
                                    f_cnt_s_o_size = f_cnt_s_o_size + 1
                                    p_b_box_pos = int(boundRect[0])  
                                    print("Relay was signaled for oversize package")
                                    
                                    
                                    
                                    
                            else:
                                print("Oversize package previously seen")
                                

                    
                starting = False

        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)        
        frame_num += 1
        image = Image.fromarray(frame)
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()
        
        

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
                


        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
        

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # Custom allowed classes (uncomment line below to allow detections for only people)
        # Use in situations where the model knows multiple classes but you only want to detect some of them.
        #allowed_classes = ['person']
        
        # For capturing good and bad images used to train the AI.
        # The images captured will not show the detection bounding boxes and labels.
        no_detect_frame = copy.deepcopy(frame)
        
        # Draw a green dot on the display so that we can get the coordinates needed for rejecting oversized packages.
        # This is a visual indication of what packages will be rejected for being too wide.
        # Green dot also mark the point at which clusters will no longer be detected as they flow from right to left across the screen.
        cv2.circle(frame, (int(640 * (1 - FLAGS.check_zone)), FLAGS.too_wide), 3, (0, 255, 0,), cv2.FILLED)         
        
        # Draw a circle above the upper left corner of the bounding rectangle at the hight which indicates to wide a package.
        # This gives a visual clue to know when a package will be rejected for being too wide.
        cv2.circle(frame, (int(boundRect[0]), FLAGS.too_wide), 3, [255,0,0], -1)   
        
        if suppress_relay == True:
            cv2.putText(frame, "Relay is suppressed.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1 , (255,0,0)) 
            cv2.putText(frame, "Press the letter e to enable relay", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1 , (255,0,0))         
        
        
        # if crop flag is enabled, crop each detection and save it as new image
        if FLAGS.crop:
            crop_rate = 150 # capture images every so many frames (ex. crop photos every 150 frames)
            crop_path = os.path.join(os.getcwd(), 'detections', 'crop', video_name)
            try:
                os.mkdir(crop_path)
            except FileExistsError:
                pass
            if frame_num % crop_rate == 0:
                final_path = os.path.join(crop_path, 'frame_' + str(frame_num))
                try:
                    os.mkdir(final_path)
                except FileExistsError:
                    pass          
                crop_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes)
            else:
                pass

        
        if FLAGS.count:
            # count objects found
            counted_classes = count_objects(pred_bbox, by_class = True, allowed_classes=allowed_classes)
             
            
            # loop through dict and print
            for key, value in counted_classes.items():
                

                # Humans only push the button to reject clusters as they pass the photo eye. 
                #The AI must do the same.
                # Only activates relay while detected clusters are on the right side of frame passing the photoeye. 
                # These are the packages coming into view.
                if pred_bbox[0][0][0] > frame_size[1] * (1 - FLAGS.check_zone):
                    print("")                        
                    print("Number of {}s: {}".format(key, value))
                    
                    
                    # Only activate relay on a cluster bounding box one time.
                    # In other words don't reject a cluster bounding box that has appeared in a previous frame.
                    if pred_bbox[0][0][0] > p_b_box_pos:
                        
                        # Prints coordinates of the bounding box around detected clusters.
                        # print("pred_bbox[0][0] is ",pred_bbox[0][0]) 
                        
                        # Puts a green dot on upper left corner of bounding box around detected clusters.
                        # Visual clue when the relay is active.
                        # cv2.circle(frame, (pred_bbox[0][0][0], pred_bbox[0][0][1]), 10, (0, 255, 0,), cv2.FILLED) 
                        
                        if FLAGS.output_images != "none":

                        
                        
                            #Save the no_detect_frame as an image in the images folder.    
                            t = datetime.datetime.now()
                            ts = t.strftime("%Y") + t.strftime("%m") + t.strftime("%d") + t.strftime("%H") + t.strftime("%M") + t.strftime("%S") + t.strftime("%f")
                            # Write the image to disk
                            cv2.imwrite(FLAGS.output_images + "/" + ts + ".jpg", no_detect_frame)
                            
                    
                            # Write an empty text file of the same root name as the image but with .txt extension. 
                            # No classes or bounding rectangle coordinates are specifed because we are telling the classifier that there is nothing to classify in this image.
                            # In fact, the reason we are taking these images is to teach the AI not to detect these circumstances.
                            # In other words, everything is well singulated.
                            f = open(FLAGS.output_images + "/" + ts + ".txt", "a")
                            f.write("")
                            f.close() 
                        


                        if using_relay == True and suppress_relay == False:
                            ser.write(bytes.fromhex("A0 01 01 A2")) # Activate the relay.
                            
                            # PLC may not see that the relay has been triggered without this.
                            time.sleep(.5) 
                            
                        
                        
                        if using_relay == True and suppress_relay == False:
                            ser.write(bytes.fromhex("A0 01 00 A1")) # Deactivate the relay
                            f_cnt_s_cluster = f_cnt_s_cluster + 1
                            print("Relay was signaled for Cluster")
                            p_b_box_pos = pred_bbox[0][0][0]   
                            
                            
                            
                    else:
                        print("Cluster previously seen")
                            
                            
                       
    

            
            image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, counted_classes, allowed_classes=allowed_classes, read_plate=FLAGS.plate) 
            # image = utils.draw_bbox(raw_frame, pred_bbox, FLAGS.info, counted_classes, allowed_classes=allowed_classes, read_plate=FLAGS.plate)            
        else:
            image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, allowed_classes=allowed_classes, read_plate=FLAGS.plate)
            # image = utils.draw_bbox(raw_frame, pred_bbox, FLAGS.info, allowed_classes=allowed_classes, read_plate=FLAGS.plate)
            f_cnt_s_cluster = 0
        
            
        # This prevents the system from rejecting packages following behind a cluster.
        if f_cnt_s_cluster == FLAGS.reset_rate:
            f_cnt_s_cluster = 0 
            p_b_box_pos = 0            
        elif f_cnt_s_cluster > 0:
            f_cnt_s_cluster = f_cnt_s_cluster + 1



            
        # This prevents the system from rejecting packages following behind an oversize package.
        if f_cnt_s_o_size == FLAGS.reset_rate:
            f_cnt_s_o_size = 0
            p_b_box_pos = 0
        elif f_cnt_s_o_size > 0:
            f_cnt_s_o_size = f_cnt_s_o_size + 1            
            
            



        fps_display = 1.0 / (time.time() - start_time)
        # print("FPS: %.2f" % fps_display)
        # result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("result", result)
            cv2.imshow("fgMask", fgMask)
            
        
        if FLAGS.output_video != "none":
            # out.write(no_detect_frame)
            # out.write(raw_frame)            
            # out.write(frame)
            out.write(result)
            
            
            
            


        keyboard = cv2.waitKey(1)
            
        if keyboard & 0xFF == ord('q'): 
            break
        
        
        elif keyboard == ord('s'):
            suppress_relay = True
            
            
        elif keyboard == ord('e'):
            suppress_relay = False            
       
        
        elif keyboard == ord('g'):
            
            if using_relay == True:
                relay_was_triggered = False
                ser.write(bytes.fromhex("A0 01 00 A1"))
                # print('Circuit Open. Button is released')
            
            
            #Save the no_detect_frame as an image in the good folder.    
            t = datetime.datetime.now()
            ts = t.strftime("%Y") + t.strftime("%m") + t.strftime("%d") + t.strftime("%H") + t.strftime("%M") + t.strftime("%S") + t.strftime("%f")
            # Write the image to disk
            cv2.imwrite(FLAGS.good_folder + "/" + ts + ".jpg", no_detect_frame)
            
    
            # Write an empty text file of the same root name as the image but with .txt extension. 
            # No classes or bounding rectangle coordinates are specifed because we are telling the classifier that there is nothing to classify in this image. 
            # In other words, everything is well singulated.
            f = open(FLAGS.good_folder + "/" + ts + ".txt", "a")
            f.write("")
            f.close()        
        
    
    
        elif keyboard == ord('b'):
    
            if anot_bboxes: # if prevents error when there is nothing to classify.
                #Save the no_detect_frame as an image in the bad folder.    
                t = datetime.datetime.now()
                ts = t.strftime("%Y") + t.strftime("%m") + t.strftime("%d") + t.strftime("%H") + t.strftime("%M") + t.strftime("%S") + t.strftime("%f")
                # Write the image to disk
                cv2.imwrite(FLAGS.bad_folder + "/" + ts + ".jpg", no_detect_frame)
                
        
                for i in anot_bboxes:
                    # Write a text file with the class index and the coordinates of the bounding rectangles. 
                    # Save this file in the Bad folder with the same root name as the image but with the .txt extension.
                    # Used to train the object classifier. 
                    # Will be class 0 Which represents Bad in the classes.txt file and represents clustered packages.
                    f = open(FLAGS.bad_folder + "/" + ts + ".txt", "a")
                    f.write("0 " + i + "\n")
                        
                f.close()      
    
            if using_relay == True:
                relay_was_triggered = True
                ser.write(bytes.fromhex("A0 01 01 A2"))
                # print('Circuit closed. Button is pressed')    
                time.sleep(.4) # Keeps the relay closed on long press. Otherwise debouce in Windows keyboard releases the relay uncommanded.
                print("")
                print("Relay activated by human")

                
        elif keyboard == -1:
            if using_relay == True:
                ser.write(bytes.fromhex("A0 01 00 A1"))
                # print('Circuit Open. Button is released')
                relay_was_triggered = False         
    

    
        anot_bboxes = [] # Empty the array for the next frame.
     
    
    
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
