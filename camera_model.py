import numpy as np

def convertLidar(scan, dataset):
    # create copy
    newScan = scan.copy()
    newScan[:, 3] = 1

    # create copy of scans to add to original after matrix modification occurs
    addScan = np.delete(scan.copy(), 3, axis=1)

    # convert xy points from lidar to camera coordinates
    newScan = np.dot(dataset.calib.T_cam2_velo, newScan.T)
    newScan = np.dot(dataset.calib.P_rect_20, newScan)
    newScan = newScan.T
    newScan = np.hstack((newScan, addScan))
    negativeMask = newScan[:, 2] > 0
    newScan = newScan[negativeMask]

    # normalize depth values
    newScan[:, 0] = newScan[:, 0] / newScan[:, 2]
    newScan[:, 1] = newScan[:, 1] / newScan[:, 2]
    #newScan[:, 2] = np.log(newScan[:, 2])

    # filter points to fit on camera images
    boundsMaskX = (newScan[:, 0] < 1241) & (newScan[:, 0] > 0)
    newScan = newScan[boundsMaskX]
    boundsMaskY = (newScan[:, 1] < 374) & (newScan[:, 1] > 0)
    newScan = newScan[boundsMaskY]

    return newScan