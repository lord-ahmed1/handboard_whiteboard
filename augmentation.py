import albumentations as A
import cv2 as cv
import os
transform=A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.8),
    A.Rotate(limit=20,p=0.5),
    A.ShiftScaleRotate(shift_limit=0.2,scale_limit=0.2)
])


i=1
imgs=os.listdir("data/masks")
for img_path in imgs:
    img=cv.imread(f"data/images/{img_path}")
    mask=cv.imread(f"data/masks/{img_path}")
    while i%50!=0:
        augmented=transform(image=img,mask=mask)
        cv.imshow("augmented_img", augmented['image'])
        cv.imshow("augmented_mask", augmented['mask'])
        cv.waitKey(1)
        cv.imwrite(f"data/images/augmented_{i}.png", augmented['image'])
        cv.imwrite(f"data/masks/augmented_{i}.png", augmented['mask'])
        i += 1
    i+=1
cv.destroyAllWindows()