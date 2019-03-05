from training.FRMW_UBAL.datapreparation.plotDataset import extractFeatures
from training.FRMW_UBAL.UbalNet import UbalNet
from training.FRMW_UBAL.UbalNet import loadModel
import os
import cv2
import numpy as np

def getRetinalReducedDataToTest(path):
    results = []
    for i in range(1, 10):
        imgpathL = path + str(i) + "l.ppm"
        imgpathR = path + str(i) + "r.ppm"

        xl, yl, sl = extractFeatures(imgpathL)
        xr, yr, sr = extractFeatures(imgpathR)
        inputRet = [xl, yl, sl, xr, yr, sr]
        results.append(inputRet)

    return results

def processImages(image):
    lower = np.array([0, 10, 0])
    upper = np.array([0, 255, 0])
    return cv2.inRange(image, lower, upper)


def blackedImages(path):

    for i in range(1, 10):
        imgpathL = path + str(i) + "l.ppm"
        imgpathR = path + str(i) + "r.ppm"
        print(imgpathL)
        imgL = cv2.imread(imgpathL)
        imgR = cv2.imread(imgpathR)

        maskL = processImages(imgL)
        maskR = processImages(imgR)

        cv2.imwrite(imgpathL, cv2.bitwise_and(imgL, imgL, mask=maskL))
        cv2.imwrite(imgpathR, cv2.bitwise_and(imgR, imgR, mask=maskR))

def testGainFields():
    path = "/home/martin/data/v2/gain_mod_testing/"
    retinalRedInf = getRetinalReducedDataToTest(path)
    model = loadModel("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/ret2")


if __name__ == '__main__':
    testGainFields()


