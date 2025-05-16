import os
import cv2 as cv
files=os.listdir("data")
import numpy as np
import json 



track_mouse=-1
def mouseCallback(event, x, y, flags, param):

    global corners
    global track_mouse
    pos= np.array([x, y])
    #when the mouse is clicked, check which corner is near cursor and assign it to track_mouse
    if event == cv.EVENT_LBUTTONDOWN:
        for index,corner in enumerate(corners):
            if np.sqrt(np.sum((pos-corner)**2))<10:
                track_mouse=index
                break #solves issue if all corners at same position
    #when the mouse is released, stop tracking
    if event == cv.EVENT_LBUTTONUP:
        track_mouse = -1
    #when the mouse is moved, move the tracked corner to the new position
    if event == cv.EVENT_MOUSEMOVE:
        if track_mouse!=-1:

            corners[track_mouse] = pos
          
        


cv.namedWindow("img")
cv.setMouseCallback("img", mouseCallback)

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


def masking(img,corners):
    h,w=img.shape[:2]
    mask=np.zeros((h,w),dtype=np.uint8)
    centerx=int(w/2)
    centery=int(h/2)
    mask[centery:centery+50,centerx:centerx+50]=255
    src_pts=np.array([[centerx,centery],[centerx+50,centery],[centerx,centery+50],[centerx+50,centery+50]],dtype=np.float32)
    homography=cv.getPerspectiveTransform(src_pts,np.array(corners,dtype=np.float32))
    mask=cv.warpPerspective(mask,homography,(w,h))
    _,mask=cv.threshold(mask,0,255,cv.THRESH_BINARY)
 
    return mask


corner_name=["top_left","top_right","bottom_left","bottom_right"]
# corners=np.zeros((4,2),dtype=np.uint16)+10

file=open("data/labels.json")
data=json.load(file)

for file in files:
    br=0
    path=f'data/{file}'
    corners=np.array(data[path],dtype=np.uint16)
    while br==0:
        img=cv.imread(path)
        mask=masking(img,corners)
        cv.imshow("mask",mask)
        masked_img=cv.bitwise_and(img,img,mask=mask.copy())
        cv.imshow("mask applied",masked_img)
        for index,corner in enumerate(corners):
                cv.circle(img, tuple(corner), 5, (10, 255, 10), -1)
                text=corner_name[index]
                cv.putText(img, text, corner+5, cv.FONT_HERSHEY_SIMPLEX, 1, (10, 255, 10), 2)
    


        cv.imshow("img",img)
        key=cv.waitKey(1)
        if key==ord("n"):
            br=1
        if key==ord("s"):
            add_new_data_field(path,corners)
            cv.imwrite(f"data/masks/{file}",mask)
