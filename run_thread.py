from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
#from imutils import resize
from mylib.mailer import Mailer
from mylib import config
from mylib.config import x1,y1,x2,y2,vertical_direction,enter_direction,cam_place
import time, schedule
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
#from itertools import zip_longest
from Lineiterator import createLineIterator
from limitexceed import check_exceed
from get_requests import send_req
from os.path import exists
from excel_data_converter import create_summary, data_converter
import queue
#import threading
from multiprocessing import Queue, Process

t0 = time.time()
class variable:
    def __init__(self):
        self.ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        self.trackers=[]
        self.trackableObjects={}
        self.totalFrames=0
        self.x=[]
        self.totalDown=0
        self.totalUp=0
        self.empty=[]
        self.empty1=[]
        self.cap=cv2.VideoCapture(config.url)
        self.H = None
        self.W= None
        self.fivemin= datetime.datetime.now()+datetime.timedelta(0,300)
        self.writer = None
        self.do_malier = 1
        tmr=datetime.datetime.now()
        try:
                #tmr=tmr.replace(day=tmr.day + 1, hour=21, minute=12, second=0, microsecond=0)
            tmr=tmr.replace(day=tmr.day + 1, hour=0, minute=0, second=0, microsecond=0)
        except ValueError:
            try:
                tmr=tmr.replace(month=tmr.month + 1, day= 1,hour=0, minute=0, second=0, microsecond=0)
            except ValueError:
                tmr=tmr.replace(year= tmr.year + 1 ,month= 1, day= 1,hour=0, minute=0, second=0, microsecond=0)
        self.tmr =tmr

var=variable()        
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=False, default = 'mobilenet_ssd/MobileNetSSD_deploy.prototxt',
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=False,default = 'mobilenet_ssd/MobileNetSSD_deploy.caffemodel',
    help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
    help="path to optional input video file")
ap.add_argument("-cam", "--camera", required=True,type=str,
    help="summary camera name")
ap.add_argument("-o", "--output", type=str,
    help="path to optional output video file")
# confidence default 0.4
ap.add_argument("-c", "--confidence", type=float, default=0.4,
    help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
    help="# of skip frames between detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# if a video path was not supplied, grab a reference to the ip camera
if not args.get("input", False):
    print("[INFO] Starting the live stream..")
    vs = VideoStream(src=config.url).start()
    #time.sleep(2.0)



# otherwise, grab a reference to the video file
else:
    print("[INFO] Starting the video..")
    vs = cv2.VideoCapture(args["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

if config.five_mins == True:
    now=datetime.datetime.now()
    fivemin= now+datetime.timedelta(0,300)
if config.people_change == True:
    peoplechangelist= []
###################################
try:
    m = ((-1*y2)-y1)/((x2)-x1)
except:
    m = 1000000001
print(m)
# m = (y2-y1)/(x2-x1)
# 0,0 -w // 2, -hi
#print(m)
iterlist=createLineIterator(np.array([int(round(x1)), int(round(y1))]),np.array([int(round(x2)), int(round(y2))]))
# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
################



#q = queue.Queue()
# start the frames per second throughput estimator
fps = FPS().start()
#if config.Thread:
# vs = thread.ThreadingClass(config.url)
def capture(q):
    while True:
        
        ret, frame = var.cap.read() # read the frames and ---
        if not ret:
            break
        if not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        q.put(frame)
        #frame = vs.read()
        
def tracker_peo(q):
    while True:
        frame = q.get()
        frame = frame[1] if args.get("input", False) else frame
        
        # if we are viewing a video and we did not grab a frame then we
        # have reached the end of the video
        if args["input"] is not None and frame is None:
            break

        # resize the frame to have a maximum width of 500 pixels (the
        # less data we have, the faster we can process it), then convert
        # the frame from BGR to RGB for dlib
        try:
            frame = imutils.resize(frame, width = 500)
        except AttributeError:
            print(frame)
            raise AttributeError


        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if the frame dimensions are empty, set them
        try:

            if var.W is None or var.H is None:
                (var.H, var.W) = frame.shape[:2]
                print(f"Frame height  is : {var.H}, frame width is : {var.W}")
        except AttributeError:
            print('(H, W) = frame.shape[:2] error')
            raise AttributeError

        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                (var.W, var.H), True)

        # initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        status = "Waiting"
        rects = []
        
        if config.five_mins == True:
            if datetime.datetime.now() >= var.fivemin:		
                enterp=info[1][1]
                exitp=info[0][1]
                send_req(enterp,exitp)
                now = datetime.datetime.now()
                var.fivemin = now + datetime.timedelta(0,300)
        if config.people_change == True:
            if len(peoplechangelist) >= 2:
                if peoplechangelist[-1] != peoplechangelist[-2]:
                    enterp=info[1][1]
                    exitp=info[0][1]
                    print(peoplechangelist)
                    send_req(enterp,exitp)
            if len(peoplechangelist) > 2:
                del peoplechangelist[:-2]
        # loop over frames from the video stream
        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if var.totalFrames % args["skip_frames"] == 0:
            # set the status and initialize our new set of object trackers
            status = "Detecting"
            var.trackers = []

            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (var.W, var.H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by requiring a minimum
                # confidence
                if confidence > args["confidence"]:
                    # extract the index of the class label from the
                    # detections list
                    idx = int(detections[0, 0, i, 1])

                    # if the class label is not a person, ignore it
                    if CLASSES[idx] != "person":
                        continue

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array([var.W, var.H, var.W, var.H])
                    (startX, startY, endX, endY) = box.astype("int")


                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    var.trackers.append(tracker)

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for tracker in var.trackers:
                # set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                status = "Tracking"

                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'
        cv2.line(frame, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (0, 0, 255), 3)
        #iterlist=createLineIterator(np.array([0, round(H * 0.50)]),np.array([W, round(H * 0.66)]),frame)
        #print(len(iterlist))
        cv2.putText(frame, "-Prediction border - Entrance-", (10, var.H - ((i * 20) + 200)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = var.ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = var.trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                if vertical_direction == 1:
                    y = [c[1] for c in to.centroids]
                    #print(y)
                    #direction = centroid[1] - np.mean(y)
                    direction =  0
                    to.centroids.append(centroid)
                    #print(to.centroids)
                    direction_all=[]
                    if len(y) >= 40:
                        # sum  of xi - mean(xi-1)
                        #try
                            #direction_all=[]
                        for index,i in enumerate(y[-41:]):
                            prev_mean= np.mean(y[:index+1])
                            direc= i - prev_mean
                            direction_all.append(direc)
                        if all([x > 0 for x in direction_all]):
                            direction = 1
                        elif all([x < 0 for x in direction_all]):    
                            direction = -1
                        else:
                            direction = 0
                        #except
                else:
                    y = [c[0] for c in to.centroids]
                    #print(y)
                    #direction = centroid[1] - np.mean(y)
                    direction =  0
                    to.centroids.append(centroid)
                    #print(to.centroids)
                    direction_all=[]
                    if len(y) >= 40:
                        # sum  of xi - mean(xi-1)
                        #try
                            #direction_all=[]
                        for index,i in enumerate(y[-41:]):
                            prev_mean= np.mean(y[:index+1])
                            direc= i - prev_mean
                            direction_all.append(direc)
                        if all([x > 0 for x in direction_all]):
                            # right
                            direction = 1
                        elif all([x < 0 for x in direction_all]):
                            #left    
                            direction = -1
                        else:
                            direction = 0

                # check to see if the object has been counted or not
                if not to.counted:
                    if centroid[0] < iterlist[0][0] or centroid[0] > iterlist[-1][0]:
                        pass
                    elif m == 1000000001 and (centroid[1] < iterlist[0][1] or centroid[1] > iterlist[-1][1]):
                        pass
                    if m < 0 and vertical_direction == 1:
                        # if the direction is negative (indicating the object
                        # is moving up) AND the centroid is above the center
                        # line, count the object
                        # H is between 0 and 500 the over the value the upper it will be, the higher the value, the lower it will be.
                        #if direction < 0 and centroid[1] < int(round(H * 0.66)):
                        #print(str(centroid))
                        if direction < 0:
                            for i in iterlist:
                                if centroid[0] > i[0] and centroid[1] < i[1]:
                                    
                                    var.totalUp += 1
                                    var.empty.append(var.totalUp)
                                    to.counted = True
                                    print('ID '+ str(to.objectID) + ' going up' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
                                    if enter_direction == 'up':
                                        check_exceed(var.x,frame)
                                    break

                        # if the direction is positive (indicating the object
                        # is moving down) AND the centroid is below the
                        # center line, count the object
                        #elif direction > 0 and centroid[1] > int(round(H * 0.66)):
                        elif direction > 0:
                            for i in iterlist:
                                if centroid[0] < i[0] and centroid[1] > i[1]:
                                    var.totalDown += 1
                                    var.empty1.append(var.totalDown)
                                    to.counted = True
                                    print('ID '+ str(to.objectID) + ' going down' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
                                    if enter_direction == 'down':
                                        check_exceed(var.x,frame)
                                    break
                                    #print(empty1[-1])
                                    # if the people limit exceeds over threshold, send an email alert
                    elif m == 0 and vertical_direction == 1:
                        
                        if direction < 0:
                            for i in iterlist:
                                if centroid[1] < i[1]:
                            
                                    var.totalUp += 1
                                    var.empty.append(var.totalUp)
                                    to.counted = True
                                    print('ID '+ str(to.objectID) + ' going up' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
                                    if enter_direction == 'up':
                                        check_exceed(var.x,frame)
                                    break

                        
                        elif direction > 0:
                            for i in iterlist:
                                if centroid[1] > i[1]:
                                    var.totalDown += 1
                                    var.empty1.append(var.totalDown)
                                    to.counted = True
                                    print('ID '+ str(to.objectID) + ' going down' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
                                    if enter_direction == 'down':
                                        check_exceed(var.x,frame)
                                    break
                                    
                    elif 0 < m < 1000000000 and vertical_direction == 1:
                        
                        if direction < 0:
                            for i in iterlist:
                                if centroid[0] < i[0] and centroid[1] < i[1]:
                            
                                    var.totalUp += 1
                                    var.empty.append(var.totalUp)
                                    to.counted = True
                                    print('ID '+ str(to.objectID) + ' going up' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
                                    if enter_direction == 'up':
                                        check_exceed(var.x,frame)
                                    break

                        
                        elif direction > 0:
                            for i in iterlist:
                                if centroid[0] > i[0] and centroid[1] > i[1]:
                                    var.totalDown += 1
                                    var.empty1.append(var.totalDown)
                                    to.counted = True
                                    print('ID '+ str(to.objectID) + ' going down' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
                                    if enter_direction == 'down':
                                        check_exceed(var.x,frame)
                                    break
                                    
                    elif m < 0 and vertical_direction == 0:
                        
                        # if the direction is negative (indicating the object
                        # is moving LEFT) AND the centroid is on the left side
                        # line, count the object
                    
                        
                        if direction < 0:
                            for i in iterlist:
                                if centroid[0] < i[0] and centroid[1] > i[1]:
                            
                                    var.totalUp += 1
                                    var.empty.append(var.totalUp)
                                    to.counted = True
                                    print('ID '+ str(to.objectID) + ' going left' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
                                    if enter_direction == 'left':
                                        check_exceed(var.x,frame)
                                    break

                        # if the direction is positive (indicating the object
                        # is moving RIGHT) AND the centroid is on the the side
                        #  line, count the object
                        elif direction > 0:
                            for i in iterlist:
                                if centroid[0] > i[0] and centroid[1] < i[1]:
                                    var.totalDown += 1
                                    var.empty1.append(var.totalDown)
                                    to.counted = True
                                    print('ID '+ str(to.objectID) + ' going right' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
                                    if enter_direction == 'right':
                                        check_exceed(var.x,frame)
                                    break
                                    
                    elif m >= 1000000000 and vertical_direction == 0:
                        # m is infinite/ vertical line
                        if direction < 0:
                            for i in iterlist:
                                if centroid[0] < i[0]:
                            
                                    var.totalUp += 1
                                    var.empty.append(var.totalUp)
                                    to.counted = True
                                    print('ID '+ str(to.objectID) + ' going left' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
                                    if enter_direction == 'left':
                                        check_exceed(var.x,frame)
                                    break

                        
                        elif direction > 0:
                            for i in iterlist:
                                if centroid[0] > i[0]:
                                    var.totalDown += 1
                                    var.empty1.append(var.totalDown)
                                    to.counted = True
                                    print('ID '+ str(to.objectID) + ' going right' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
                                    if enter_direction == 'right':
                                        check_exceed(var.x,frame)
                                    break
                                    
                    elif 0 < m < 1000000000 and vertical_direction == 0:
                        
                        if direction < 0:
                            for i in iterlist:
                                if centroid[0] < i[0] and centroid[1] < i[1]:
                            
                                    var.totalUp += 1
                                    var.empty.append(var.totalUp)
                                    to.counted = True
                                    print('ID '+ str(to.objectID) + ' going left' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
                                    if enter_direction == 'left':
                                        check_exceed(var.x,frame)
                                    break

                        
                        elif direction > 0:
                            for i in iterlist:
                                if centroid[0] > i[0] and centroid[1] > i[1]:
                                    var.totalDown += 1
                                    var.empty1.append(var.totalDown)
                                    to.counted = True
                                    print('ID '+ str(to.objectID) + ' going right' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
                                    if enter_direction == 'right':
                                        check_exceed(var.x,frame)
                                    break
                                        
                    var.x = []
                    # compute the sum of total people inside
                    if enter_direction == 'down' or enter_direction == 'right':
                        var.x.append(len(var.empty1)-len(var.empty))
                    else:
                        var.x.append(len(var.empty)-len(var.empty1))
                    #print("Total people inside:", x)


            # store the trackable object in our dictionary
            var.trackableObjects[objectID] = to
            
            #print(peoplechangelist)
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        # construct a tuple of information we will be displaying on the
        if enter_direction == 'down' or enter_direction == 'right':
            info = [
            ("Exit", var.totalUp),
            ("Enter", var.totalDown),
            ("Status", status),
            ]
        else:
            info = [
            ("Exit", var.totalDown),
            ("Enter", var.totalUp),
            ("Status", status),
            ]

        info2 = [
        ("Total people inside", var.x),
        ]
        if config.people_change == True:
                        #if len(peoplechangelist) > 0:
            peoplechangelist.append(var.x)
        
        try:
            assert objectID
            assert centroid
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
        except (AssertionError,ValueError,NameError):
            pass
        try:
            if int(info2[0][1][0]) >= config.Threshold:
                cv2.putText(frame, "-ALERT: People limit exceeded-", (10, frame.shape[0] - 80),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
                if config.ALERT:
                    if var.do_malier == 1:
                        print("[INFO] Sending email alert..")
                        try:
                            Mailer().send(config.MAIL)
                        except:
                            var.do_malier = 0
                        print("[INFO] Alert sent")
        except IndexError:
            pass	
        # Display the output
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, var.H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        for (i, (k, v)) in enumerate(info2):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (265, var.H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Initiate a simple log to save data at end of the day
        if config.Log:
            try:
                timeinxmins
            except NameError:
                timeinxmins=datetime.datetime.now() + config.timedel

            if datetime.datetime.now() >= timeinxmins:
                #data={'櫃位地點':config.cam_place,'People Enter':info[1][1],'People Exit':info[0][1],'Current People Inside':info2[0][1],'Date':datetime.datetime.now()}
                #df=pd.DataFrame(data=data)
                timeinxmins=datetime.datetime.now() + config.timedel
                cam_place=str(args["camera"])
                excel_name=f"./summary/{cam_place} summary.xlsx"
                if exists(excel_name):
                    #with pd.ExcelWriter(excel_name,mode='a')  as writer:
                    #append_df_to_excel(excel_name, df,header=None, index=False)
                    data_converter(info[1][1],info[0][1],excel_name)  
                else:
                    create_summary(info[1][1],info[0][1],excel_name)
                print('summary exported!')
                
        # check to see if we should write the frame to disk
        if var.writer is not None:
            writer.write(frame)

        # show the output frame
        cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        var.totalFrames += 1
        fps.update()

        if config.Timer:
            # Automatic timer to stop the live stream. Set to 8 hours (28800s).
            t1 = time.time()
            num_seconds=(t1-t0)
            if num_seconds > 28800:
                break
        if config.Scheduler:
                if datetime.datetime.now() >= var.tmr :
                    print('renew program')
                    raise ValueError
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


    # # if we are not using a video file, stop the camera video stream
    # if not args.get("input", False):
    # 	vs.stop()
    #
    # # otherwise, release the video file pointer
    # else:
    # 	vs.release()
    
    # issue 15
    #if config.Thread:
        #vs.release()

    # close any open windows
    cv2.destroyAllWindows()

def start_thread():
    q= Queue()
    cam_process = Process(target=capture, args=(q,))
    cam_process.start()
    think_process = Process(target=tracker_peo, args=(q,))
    think_process.start()
    cam_process.join()
    think_process.join()
##learn more about different schedules here: https://pypi.org/project/schedule/
if __name__ == '__main__':

    q= Queue()
    if config.Scheduler:
        ##Runs for every 1 second
        global tmr
        tmr=datetime.datetime.now()
        schedule.every(1).seconds.do(start_thread)
        
        
        ##Runs at every day (9:00 am). You can change it.
        #schedule.every().day.at("9:00").do(run)
        while 1:
            var.tmr=datetime.datetime.now()
            try:
                #tmr=tmr.replace(day=tmr.day + 1, hour=21, minute=12, second=0, microsecond=0)
                var.tmr=var.tmr.replace(day=var.tmr.day + 1, hour=0, minute=0, second=0, microsecond=0)
            except ValueError:
                try:
                    var.tmr=var.tmr.replace(month=var.tmr.month + 1, day= 1,hour=0, minute=0, second=0, microsecond=0)
                except ValueError:
                    var.tmr=var.tmr.replace(year= var.tmr.year + 1 ,month= 1, day= 1,hour=0, minute=0, second=0, microsecond=0)
            #print(tmr)
            #print(datetime.datetime.now())
            #try:
            schedule.run_pending()

            if datetime.datetime.now() >= var.tmr:
                print('renew program')
                raise ValueError
                    
            #except:
                #print('schedule error')
                #continue
    #else:
        #run()