import cv2 as cv
import time
import numpy as np
from ultralytics import YOLO
import json
import datetime
from time import sleep
from queue import Queue
from threading import Thread
import os


frame_queue = Queue(maxsize=1)
display_width,display_height=1920,1080


def recorderThread(frame_queue):  #to achieve more accurate recording and not affected by code delays
    global  record_mode 
    print("recorder thread started")
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    videoWriter = cv.VideoWriter(f'recordings/{current_time}.mp4', fourcc,2, (display_width,  display_height))
    saved_frame=np.zeros((display_height,display_width),dtype=np.uint8)

    while record_mode==1:
        if frame_queue.full():
            frame = frame_queue.get()
            saved_frame=frame
        else:
            frame=saved_frame
        videoWriter.write(cv.cvtColor(frame.copy(),cv.COLOR_GRAY2BGR))
        sleep(1/2)
    videoWriter.release()


track_mouse=-1
def mouseCallback(event, x, y, flags, param):
    print("mouse callback")
    global corners
    global track_mouse
    pos = np.array([x, y])
    print(pos,"pos")
    #when the mouse is clicked, check which corner is near cursor and assign it to track_mouse
    if event == cv.EVENT_LBUTTONDOWN:
        print("mouse down")
        for corner in corners:
            corner_value=corners[corner]
            print(corner,corner_value,"values")
            print(pos,"pos")

            if np.sqrt(np.sum((pos-corner_value)**2))<20:
                track_mouse=corner
                break #solves issue if all corners at same position
    #when the mouse is released, stop tracking
    if event == cv.EVENT_LBUTTONUP:
        track_mouse = -1
    #when the mouse is moved, move the tracked corner to the new position
    if event == cv.EVENT_MOUSEMOVE:
        if track_mouse!=-1:
            corners[track_mouse] = pos
          
        
def obtain_and_remove(corners,index):
    #obtain the corner at the index and remove it from the corners
    corner=corners[index]
    corners=corners[np.where(corners!=corner)[0]]
    return corner,corners
def sort_corners(corners,img):
    h,w=img.shape[:2]
    #measure the distance from each corner to the four corners of the image
    distances_from_top_left=np.sum(corners**2,axis=1)
 
    #obtain the index of the corner with the minimum distance from each corner
    top_left_index     = np.where(distances_from_top_left==np.min(distances_from_top_left))
    top_left,corners=obtain_and_remove(corners,top_left_index[0][0])

   
    distances_from_top_right=np.sum((corners-[w,0])**2,axis=1)
    top_right_index    = np.where(distances_from_top_right==np.min(distances_from_top_right))
    top_right,corners=obtain_and_remove(corners,top_right_index[0][0])

    distances_from_bottom_right=np.sum((corners-[w,h])**2,axis=1)
    bottom_right_index =np.where(distances_from_bottom_right==np.min(distances_from_bottom_right))
    bottom_right,corners=obtain_and_remove(corners,bottom_right_index[0][0])

   

    distances_from_bottom_left=np.sum((corners-[0,h])**2,axis=1)
    bottom_left_index  =np.where(distances_from_bottom_left==np.min(distances_from_bottom_left))
    bottom_left,corners=obtain_and_remove(corners,bottom_left_index[0][0])


    return {"top_left":top_left,"top_right":top_right,"bottom_left":bottom_left,"bottom_right":bottom_right}

def filter(img):
    h,w=img.shape[:2]
    scalex,scaley=int(display_width/w),int(display_height/h)
    scale_array=np.array([scalex,scaley])
    img_corners=np.float32([[0,0],[w,0],[0,h],[w,h]])*scale_array
    corners_array=np.float32([corners["top_left"],corners["top_right"],corners["bottom_left"],corners["bottom_right"]])
    matrix,_=cv.findHomography(corners_array*scale_array,img_corners)
    canny=cv.Canny(cv.resize(img,([w,h]*scale_array)),130,140)
    canny=cv.dilate(canny,(2,2),iterations=1)
    for index,corner in enumerate(corners):
        corner_value=corners[corner]
        cv.circle(img, tuple(corner_value), 4, (255, 255, 255), -1)
        
        cv.putText(img, corner, corner_value+15, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv .imshow("enhanced",img)
    try:
        canny=cv.warpPerspective(canny, matrix, (display_width,display_height))
    except:
        pass
    
    return canny

url = "http://192.168.1.5:4747/video"
url = 2

cap = cv.VideoCapture(url)

if not cap.isOpened():
    print("Failed to connect to DroidCam stream.")
    exit()


br=0
imageN=len(os.listdir('data'))

record_mode=-1
first_iteration=0

def add_new_data_field(image_path,corners):
    try:
        file=open('data/labels.json','r')
        j_object=json.load(file)
        file.close()
    except:
        j_object={}
    file=open('data/labels.json','w')
    j_object[image_path] = {key: value.tolist() for key, value in corners.items()}

    j_object=json.dumps(j_object)
    file.write(j_object)


show=np.zeros((display_width,display_height,3),dtype=np.uint8)



model = YOLO("s.pt")  # Replace "s.pt" with your trained YOLO model file
model.overrides['verbose'] = False


corners={"top_left":np.array([50,50]),"top_right":np.array([50,50]),"bottom_left":np.array([50,50]),"bottom_right":np.array([50,50])}

cv.namedWindow("enhanced")
cv.setMouseCallback("enhanced", mouseCallback)


while br==0:
    ret, frame = cap.read()
    results = model(frame,conf=0.5)
  

    for result in results:
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()


            # Convert mask to binary
            mask_uint8 = (masks[0] * 255).astype(np.uint8)
            

            # Find contours
            contours, _ = cv.findContours(mask_uint8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours_sizes= [(cv.contourArea(cnt), cnt) for cnt in contours]
            contour = max(contours_sizes, key=lambda x: x[0])[1]

            for contourn in contours:
            

                # Approximate contour to polygon
                epsilon = 0.02 * cv.arcLength(contour, True)
                approx = cv.approxPolyDP(contour, epsilon, True)
                # cv.drawContours(frame,[approx],0,(255,0,255),2)


                obtained_corners = approx.reshape(-1, 2)
                corners=sort_corners(obtained_corners,frame)

                # Draw the corners on the image
                for point in corners:
                    corner_value=corners[point]
                    cv.circle(frame, tuple(corner_value), 5, (0, 0, 255), -1)

    if ret:
        show=filter(frame)
        cv.imshow("DroidCam Stream", show)
    key=cv.waitKey(1)
    if key==ord("q"):
        record_mode=0
        br=1
    if key==ord("d"):
        print("saving")
        cv.imwrite(f'data/images/{imageN}.jpg',frame)
        add_new_data_field(f'data/{imageN}.jpg',corners)
        imageN+=1
    if key==ord("s"):
        print("saving")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cv.imwrite(f'screen_shots/{current_time}.jpg', show)
        print("saved")
    if key==ord("r"):
        record_mode*=-1
        if record_mode==1 :
            Thread(target=recorderThread, daemon=False,args=(frame_queue,)).start()

    if record_mode == 1 and not frame_queue.full():
        frame_queue.put(show.copy())



   
record_mode=0
cap.release()
print("done")

