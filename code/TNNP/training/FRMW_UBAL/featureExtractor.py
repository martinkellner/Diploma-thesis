
import os
import numpy as np
import cv2
import  pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


from UbalNet import *

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

def markIncorrectSamples(filename, filterThreshold=50):
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

def extractFeatures(path):
    img = cv2.imread(path)

    # convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert the grayscale image to binary image

    ret, thresh = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)
    size = sum(1 for i in range(len(thresh)) for j in range(len(thresh[i])) if thresh[i][j] != 0)
    # find contours in the binary image
    '''for i in range(len(thresh)):
        for j in range(len(thresh[i])):
            print(thresh[i][j], end=' ')
        print()
    '''
    M = cv2.moments(thresh)
    # calculate x,y coordinate of center

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    #cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
    return cX, cY, size

def image():
    img = cv2.imread("/home/martin/data/v2/140_l_img.ppm")
    img2 = cv2.imread("/home/martin/data/v2/filtered/140_l_img_blc.ppm")

    x,y,s = extractFeatures("/home/martin/data/v2/filtered/140_l_img_blc.ppm")
    print(x, y ,s)

    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(img[:,:,::-1])
    axes[1].imshow(img2[:,:,::-1])

    axes[2].imshow(img2[:, :, ::-1])
    axes[2].scatter(x, y, marker='o', color='red', s=5 )

    axes[0].axis("off")
    axes[1].axis("off")

    plt.show()
    plt.savefig("/home/martin/data.png")


def addFeaturesToFilteredDataset(path):
    skipped = 0
    path = path + '/' if path[-1] is not '/' else path
    with open(path + "correct.txt", 'r') as infile:
        with open(path + "new2.txt", 'w') as oufile:
            i = 0
            for line in infile:

                if line[0] != '#':

                    xr, yr, szr = extractFeatures(path + line[0:line.find(" ")] + "_r_img_blc.ppm")
                    xl, yl, szl = extractFeatures(path + line[0:line.find(" ")] + "_l_img_blc.ppm")

                    if xr is None or xl is None:
                        print("Skipping line!")
                        skipped += 1
                        continue

                    #line = line[:-1] + str(xr) + " " + str(yr) + " " + str(szr) + " " + str(xl) + " " + str(yl) + " " + str(szl)
                    line = line[:-1] + path + line[0:line.find(" ")] + "_r_img_blc.ppm" + " " + path + line[0:line.find(" ")] + "_l_img_blc.ppm"
                    print(line, file=oufile)
                    print("{}. line - Done!".format(i))
                i += 1
    print("{} lines skipped!".format(skipped))


def printFixPoints(path):
    df = pd.read_csv(path)
    newdf = df[["FX", "FY", "FZ"]]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    n = 100

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    xs = newdf["FX"].values
    ys = newdf["FY"].values
    zs = newdf["FZ"].values
    ax.scatter(xs, ys, zs)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    print(newdf)

def filerSamplesOverErrorLimit(errorLim=2):
    '''
    Filter out samples with errors over the limit
    '''
    dataset = pd.read_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/retinal_image/dataset.csv", sep=',', dtype={'E1': float, 'E2': float, 'E3': float})
    filtered = dataset[(dataset.E1 < errorLim) & (dataset.E2 < errorLim) & (dataset.E3 < errorLim)]
    filtered.to_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/retinal_image/dataset_flt.csv")

def selectSpecificSamples():
    df = pd.read_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/dataset_ret.csv")
    filtered = df[(df.FX < -.165)]
    filtered.to_csv("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/dataset_ret_add.csv")

if __name__ == '__main__':
    #extractFeatures("/home/martin/data/filtered/10_r_img_blc.ppm")
    #addFeaturesToFilteredDataset()
    #printFixPoints()

    # The second collected dataset for ret model!
    #markIncorrectSamples("/home/martin/data/v2/dataset.txt")
    #addFeaturesToFilteredDataset("/home/martin/data/v2/filtered/")
    #filerSamplesOverErrorLimit(2)   ### -> new filtered dataset
    #### DEPRECATED ### selectSpecificSamples() #-> addind data from old dataset to the new dataset
    #printFixPoints("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/datapreparation/data/retinal_image/dataset_flt.csv")
    pass
    image()