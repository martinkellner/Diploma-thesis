
import os
import numpy as np
import cv2

from training.FRMW_UBAL import UbalNet

#def readFile(path):
#    return pd.read_csv(path, sep=',')

def checkImagesGreenPixel(foldername, prefix, threshold, show=False) -> bool:
    """
    Check if images contains green object, if it does the green area will be seperated.
    Example of name convention for images: 0_r_img.ppm.
    """
    image01 = cv2.imread(foldername + prefix + "_r_img.ppm")
    image02 = cv2.imread(foldername + prefix + "_l_img.ppm")
    images = [image01, image02]
    output = [None, None]

    lower = np.array([0, 10, 0])
    upper = np.array([0, 255, 0])
    result = True

    for i in range(2):
        mask = cv2.inRange(images[i], lower, upper)
        count = countNULLValuesInMap(mask)
        print(prefix + ":" + str(count))
        if count < threshold:
            result = False
        output[i] = cv2.bitwise_and(images[i], images[i], mask=mask)
        if show:
            cv2.imshow("images", np.hstack([images[i], output]))
            cv2.waitKey(0)

    cv2.imwrite(foldername + "filtered/" + prefix + "_r_img_blc.ppm", output[0])
    cv2.imwrite(foldername + "filtered/" + prefix + "_l_img_blc.ppm", output[1])
    return result

def countNULLValuesInMap(map) -> int:
    """
    Count up the number of null pixels and returns it.
    """
    count = 0
    for i in range(len(map)):
        for j in range(len(map[i])):
            if map[i][j] != 0:
                count += 1
    return count

def markIncorrectSamples(filename, filterThreshold=250):
    """
    Check if images of sample contains green object.
    If one of images is incorrect the line in dataset file is marked with '#' char at start of the line.
    """
    file = None
    if os.path.exists(filename):
        file = open(filename, 'r')
    else:
        print(format("EXIT: File {} does not exist", filename))
        return

    foldername = filename[0:filename.rfind("/")+1]
    if not os.path.exists(foldername + "filtered/"):
        os.mkdir(foldername + "filtered/")
    crrtfile = open(foldername + "filtered/" + "correct.txt", 'w+')
    line = file.readline()

    while line:
        if line[0] != "#":
            correct = checkImagesGreenPixel(foldername, line[0:line.find(" ")], threshold=filterThreshold)
            if not correct:
                line = "#" + line

        crrtfile.write(line)
        line = file.readline()

    file.close()
    crrtfile.close()

if __name__ == '__main__':
    #markIncorrectSamples("/home/martin/data/dataset.txt", filterThreshold=250)

    X = np.array([[1, 0, 0, 0, 0, 1, 0, 1],
                  [0, 1, 0, 0, 0, 0, 1, 1],
                  [0, 0, 1, 0, 1, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0]])
    y = np.array([[1, 0, 0, 0, 1, 1],
                  [0, 1, 0, 0, 1, 0],
                  [0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 1, 1]])

    un1 = UbalNet()
    un1.setHyperparamenters(.1, 100, 10)
    un1.fit(X, y)