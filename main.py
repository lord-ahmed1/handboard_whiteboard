import cv2 as cv
import time
import numpy as np
from ultralytics import YOLO
import json


model = YOLO("seg.pt")  # Replace "best.pt" with your trained YOLO model file
model.overrides['verbose'] = False

corners=np.array([[50,50],[50,50],[50,50],[50,50]])
corner_name=["top_left","top_right","bottom_left","bottom_right"]

track_mouse=-1
def mouseCallback(event, x, y, flags, param):
    global corners
    global track_mouse
    pos = np.array([x, y])
    #when the mouse is clicked, check which corner is near cursor and assign it to track_mouse
    if event == cv.EVENT_LBUTTONDOWN:
        for index,corner in enumerate(corners):

            if np.sqrt(np.sum((pos-corner)**2))<20:
                track_mouse=index
                break #solves issue if all corners at same position
    #when the mouse is released, stop tracking
    if event == cv.EVENT_LBUTTONUP:
        track_mouse = -1
    #when the mouse is moved, move the tracked corner to the new position
    if event == cv.EVENT_MOUSEMOVE:
        if track_mouse!=-1:
            corners[track_mouse] = pos
          
        

def sort_corners(corners):
    """
    corners: numpy array of shape (4, 2)
    returns: sorted corners in order:
             top-left, top-right, bottom-left, bottom-right
    """
    # Ensure float
    corners = np.array(corners, dtype="float32")

    # Sum and diff of points
    s = corners.sum(axis=1)         # x + y
    diff = np.diff(corners, axis=1) # y - x

    top_left     = corners[np.argmin(s)]
    bottom_right = corners[np.argmax(s)]
    top_right    = corners[np.argmin(diff)]
    bottom_left  = corners[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_left, bottom_right],dtype=np.uint16)

cv.namedWindow("enhanced")
cv.setMouseCallback("enhanced", mouseCallback)

sharpen_kernel = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]])

def filter(img):
    h,w=img.shape[:2]

    img_corners=np.float32([[0,0],[w,0],[0,h],[w,h]])*2
    matrix,_=cv.findHomography(np.float32(corners)*2,img_corners)
   

    blurred = cv.GaussianBlur(img, (11, 11), 0)


    # Enhance details from the second image
    img = cv.addWeighted(img, 1.5, blurred, -1, 0)

    img=cv.resize(img,(w*2,h*2))


   

    # Apply sharpening filter
    canny=cv.Canny(img,130,140)
    canny = cv.filter2D(canny, -1, sharpen_kernel)



    # canny=cv.resize(canny,(w*2,h*2))


    # for index,corner in enumerate(corners):
    #     cv.circle(img, tuple(corner), 4, (255, 255, 255), -1)
    #     text=corner_name[index]
    #     cv.putText(img, text, corner+15, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    try:
        output_size = (int(img_corners[3][0]), int(img_corners[3][1]))  # Set output size based on img_corners
        canny=cv.warpPerspective(canny, matrix, output_size)
    except:
        pass
    
    return canny

url = "http://192.168.1.5:4747/video"
url = 0

cap = cv.VideoCapture(url)

if not cap.isOpened():
    print("Failed to connect to DroidCam stream.")
    exit()


br=0
import os
imageN=len(os.listdir('data'))

def add_new_data_field(image_path,corners):
    try:
        file=open('data/labels.json','r')
        j_object=json.load(file)
        file.close()
    except:
        j_object={}
    file=open('data/labels.json','w')
    j_object[image_path]=corners.tolist()

    j_object=json.dumps(j_object)
    file.write(j_object)

while br==0:
    ret, frame = cap.read()
    results = model(frame,conf=0.2)
    # for result in results:
    #     boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
    #     confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
    #     try:
    #         max_conf=np.max(confidences)
    #         conf_index=np.where(confidences==max_conf)[0][0]

    #         box=boxes[conf_index]
    #         x1, y1, x2, y2 = map(int, box)
    #         corners=np.array([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])*2
    #         cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
    #     except:
    #         pass


    result=results[0]
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        max_conf=np.max(confs)
        conf_index=np.where(confs==max_conf)[0][0]
        mask=masks[conf_index]


        # Convert mask to binary
        mask_uint8 = (mask * 255).astype(np.uint8)
        

        # Find contours
        contours, _ = cv.findContours(mask_uint8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:  # Likely a rectangle (like paper)
                corners = approx.reshape(4, 2)
                corners=sort_corners(corners)
                print("📌 Paper corners:", corners)

                # Draw the corners on the image
                for point in corners:
                    cv.circle(frame, tuple(point), 5, (0, 0, 255), -1)
    cv.imshow("frame",frame)



   

  
    if ret:
        show=filter(frame)
        cv.imshow("DroidCam Stream", show)
    key=cv.waitKey(1)
    if key==ord("q"):
        br=1
    if key==ord("s"):
        print("saving")
        cv.imwrite(f'data/{imageN}.jpg',frame)
        add_new_data_field(f'data/{imageN}.jpg',corners)
        imageN+=1


    

