from ultralytics import YOLO
import pykitti
import numpy as np
import torch
import cv2
from camera_model import convertLidar
import time

startTime = time.perf_counter()

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
confThreshold = 0.4

#all image generators
imgGen = dataset.cam2
veloGen = dataset.velo

#labels of objects
classLabels = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

kInverse = np.linalg.inv(dataset.calib.K_cam2)
rtInverse = np.linalg.inv(dataset.calib.T_cam2_velo)

for img, velo in zip(imgGen, veloGen):
    #get yolo prediction
    results = model.predict(img)

    #convert img to opencv friendly format
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    scanData = convertLidar(velo, dataset)

    #create mask and filter
    cls = results[0].boxes.cls
    conf = results[0].boxes.conf
    mask = torch.isin(cls, torch.tensor(classIDArr, device=cls.device)) & (conf > confThreshold)
    filteredResults = results[0].boxes[mask]

    #convert to numpy array
    rect = filteredResults.xyxy.numpy()
    label = filteredResults.cls.numpy()
    confidence = filteredResults.conf.numpy()

    #normalize and create colormap
    depthNorm = cv2.normalize(scanData[:, 2],None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    depthNorm = 255 - depthNorm
    allColors = cv2.applyColorMap(depthNorm.reshape(-1, 1), cv2.COLORMAP_JET)

    #create bounding box, label, and confidence level in each image
    for r, l, c in zip(rect, label, confidence):
        cv2.rectangle(img, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (0, 255, 0), 1)
        cv2.putText(img, classLabels[l], (int(r[0]), int(r[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)

        boundBoxMask = (scanData[:,0] > int(r[0])) & (scanData[:,0] < int(r[2])) & (scanData[:,1] > int(r[1])) & (scanData[:,1] < int(r[3]))
        boxData = scanData[boundBoxMask].copy()
        colors = allColors[boundBoxMask].copy()
        if len(boxData) == 0:
            continue
        depthValues = boxData[:,2].ravel()
        quartiles = np.quantile(depthValues, [0.25, 0.5, 0.75])

        #back projection
        u = (r[0] + r[2]) / 2
        v = (r[1] + r[3]) / 2
        imgVector = np.array([u, v, 1])
        imgVector = imgVector.reshape(-1, 1)
        backProjectCam = quartiles[0] * np.dot(kInverse, imgVector)
        backProjectCam = np.vstack((backProjectCam, 1))
        backProjectLidar = np.dot(rtInverse, backProjectCam)
        distance = np.linalg.norm(backProjectLidar)
        cv2.putText(img, str(round(distance, 1)), (int(r[0]), int(r[3] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 2)

        #draw lidar point overlaid on objects
        for p, c in zip(boxData, colors):
            color = tuple(map(int, c[0]))
            cv2.circle(img, (int(p[0]), int(p[1])), 1, color, -1)


    #display image
    video.write(img)

video.release()
cv2.destroyAllWindows()

endTime = time.perf_counter()
print(f"Program runtime: {endTime - startTime:.2f} seconds")