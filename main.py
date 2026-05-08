from ultralytics import YOLO
import pykitti
import numpy as np
import torch
import cv2

# Load the dataset
basedir = './data/kitti'
date = '2011_09_26'
drive = '0015'

#create pykitti dataset
dataset = pykitti.raw(basedir, date, drive)

#create yolo model
model = YOLO('yolo11n.pt')

#create opencv videowriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('fullDrive.mp4', fourcc, 10, (1242, 375))

#filter thresholds
classIDArr = [0, 1, 2, 3, 5, 6, 7]
confThreshold = 0.6

#all image generators
imgGen = dataset.cam2

#labels of objects
classLabels = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

for img in imgGen:
    #get yolo prediction
    results = model.predict(img)

    #convert img to opencv friendly format
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    #create mask and filter
    cls = results[0].boxes.cls
    conf = results[0].boxes.conf
    mask = torch.isin(cls, torch.tensor(classIDArr, device=cls.device)) & (conf > confThreshold)
    filteredResults = results[0].boxes[mask]

    #convert to numpy array
    rect = filteredResults.xyxy.numpy()
    label = filteredResults.cls.numpy()
    confidence = filteredResults.conf.numpy()

    #create bounding box, label, and confidence level in each image
    for r, l, c in zip(rect, label, confidence):
        cv2.rectangle(img, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (0, 255, 0), 3)
        cv2.putText(img, classLabels[l], (int(r[0]), int(r[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.putText(img, str(c), (int(r[2]), int(r[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    #display image

    video.write(img)

    '''
    cv2.imshow('image', img)
    cv2.waitKey(10)
    cv2.destroyAllWindows()
    '''

video.release()
cv2.destroyAllWindows()