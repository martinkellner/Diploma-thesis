import numpy as np
import seaborn as sns
import os
import cv2
import matplotlib.pyplot as plt

def processImagesToMap(path):
    if os.path.exists(path):

        countAll = 0
        countCntObj = 0
        path = path + '/' if path[-1] == '/' else path
        files = [file for _, _, file in os.walk(path)]

        lower = np.array([0, 10, 0])
        upper = np.array([0, 255, 0])

        rMap = np.zeros(shape=(240, 320))
        lMap = np.zeros(shape=(240, 320))

        for i in range(280):
            rFile = path + str(i+1) + "r_img.ppm"
            lFile = path + str(i+1) + "l_img.ppm"
            if os.path.exists(rFile) and os.path.exists(lFile):
                imgR = cv2.imread(rFile)
                imgL = cv2.imread(lFile)

                rMask = cv2.inRange(imgR, lower, upper)
                lMask = cv2.inRange(imgL, lower, upper)

                rMap = rMap + rMask
                lMap = lMap + lMask

        plt.cla()
        plt.close()
        #fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True, sharey=True)

        #cbar_ax = fig.add_axes([1, .3, .03, .4])
        sns.heatmap(rMap, vmax=12000)
        plt.savefig("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/vizualizer/plots/right_eye_hm4.png")
        plt.close()
        sns.heatmap(lMap, vmax=12000)
        plt.savefig("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/vizualizer/plots/left_eye_hm4.png")
        #plt.show()

if __name__ == '__main__':
    processImagesToMap("/home/martin/data/newTesting/2-1v2np/")