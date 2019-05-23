from featureExtractor import extractFeatures
from UbalNet import loadModel
from sklearn.externals import joblib
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from DataPreparation import inputCoder

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
        #print(imgpathL)
        imgL = cv2.imread(imgpathL)
        imgR = cv2.imread(imgpathR)

        maskL = processImages(imgL)
        maskR = processImages(imgR)

        cv2.imwrite(imgpathL, cv2.bitwise_and(imgL, imgL, mask=maskL))
        cv2.imwrite(imgpathR, cv2.bitwise_and(imgR, imgR, mask=maskR))

def testGainFields(imagespath, neuron=1, version=None):
    retinalRedInf = getRetinalReducedDataToTest(imagespath)
    tilts = [9.8, 0, -10, -19]
    versions = [-29, -15, 0, 15, 29]
    vergence = 30

    loadScaler = joblib.load("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/scaler/sc5.pkl")
    model = loadModel('/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/2-1v2np/')
    print(model.getName())
    #0.1_20_200_0.2_0.9_2019-04-10_19:13:36
    idx = 1
    #path = pathToSave + '/' if pathToSave[-1] != '/' else pathToSave
    resultActs = list()
    emResultActs = list()

    for i in range(len(retinalRedInf)):
        activationForTitlAndVers = []
        emActivationForTitlAndVers = []
        for titlIdx in range(len(tilts)):
            for versionIdx in range(len(versions)):
                if version is None:
                    inputNonScaled = [0] * 16
                    inputNonScaled[0] = tilts[titlIdx]
                    inputNonScaled[1] = versions[versionIdx]
                    inputNonScaled[2] = vergence

                    for j in range(3):
                        inputNonScaled[3+j] = retinalRedInf[i][j]
                    for k in range(3):
                        inputNonScaled[6+k] = retinalRedInf[i][k+3]

                    inputNonScaled[5] = 600
                    inputNonScaled[8] = 600

                    #Scale created input with the scaler used for testing data
                    scaledInput = loadScaler.transform([inputNonScaled])
                    x = scaledInput[0][:9]

                    # Run forward predition on the best model
                    model.predictForward(x)
                    hiddenActivations = model.get_last_activations()[neuron]

                    x[0] = 0
                    x[1] = 0
                    x[2] = 0
                    model.predictForward(x)
                    emphiddenActivations = model.get_last_activations()[neuron]

                    activationForTitlAndVers.append(hiddenActivations)
                    emActivationForTitlAndVers.append(emphiddenActivations)

                else:
                    inpNeurons = [10, 12, 8, 32, 28, 32, 28]
                    inpWidths = [4, 5, 4, 8, 6, 8, 6]

                    inputNonCoded = [tilts[titlIdx], versions[versionIdx], vergence, retinalRedInf[i][0], retinalRedInf[i][1], 600, retinalRedInf[i][3], retinalRedInf[i][4], 600]
                    titlI, versionI, vergenceI, x1I, y1I, x2I, y2I = inputCoder(inputNonCoded, inpNeurons, inpWidths)
                    encodedInput = titlI + versionI + vergenceI + x1I + y1I + x2I + y2I

                    model.predictForward(encodedInput)
                    hiddenActivations = model.get_last_activations()[neuron]

                    for inx in range(0, 30):
                        encodedInput[inx] = 0
                    
                    model.predictForward(encodedInput)
                    emphiddenActivations = model.get_last_activations()[neuron]
    
                    activationForTitlAndVers.append(hiddenActivations)
                    emActivationForTitlAndVers.append(emphiddenActivations)

        resultActs.append(activationForTitlAndVers)
        emResultActs.append(emActivationForTitlAndVers)

    return np.array(resultActs), np.array(emResultActs)

def plotGainModulations(results, neuron):
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    fig.subplots_adjust(.05, .05, .95, .95)
    rcParams['axes.labelcolor'] = 'red'
    diameter = 700
    xscl = [-30, -15, 0, 15, 30]
    yscl = [10, 0, -10, -20]
    pointsX = []
    pointsY = []
    sizes = []
    titles = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

    for y in yscl:
        for x in xscl:
            pointsX.append(x)
            pointsY.append(y)
            sizes.append(diameter)

    for i in range(3):
        for j in range(3):
            ax[i][j].scatter(pointsX, pointsY, s=sizes, facecolors='none', edgecolors='g')
            ax[i][j].set_xlim([-40, 40])
            ax[i][j].set_ylim([-25, 10])

            ax[i][j].spines['right'].set_visible(False)
            ax[i][j].spines['top'].set_visible(False)

            # Only show ticks on the left and bottom spines
            ax[i][j].yaxis.set_ticks_position('left')
            ax[i][j].xaxis.set_ticks_position('bottom')
            plt.setp(ax, xticks=xscl, yticks=yscl) #, yticklabels=["-20", "-10", "0", "10"]

    for s1 in range(results.shape[0]):
        sizesTotal = []
        sizesEmp   = []
        maxResponse = 1 #max( max(results[:,s1, 0]), max(results[:,s1, 1]))
        excRepsIndex = []
        #print(maxResponse)

        for s0 in range(results.shape[1]):
            #print(s1*20,(s1+1)*20)
            sizesTotal.append((results[s1, s0, 0]/maxResponse)*diameter)
            exc = True if results[s1, s0, 1] < 0.0 else False
            emp = (results[s1, s0, 0] - results[s1, s0, 1] if exc else results[s1, s0, 1])
            sizesEmp.append((emp/maxResponse)*diameter)

            if exc:
                excRepsIndex.append(s0)

        #print(sizesTotal)
        #print(pointsX)

        excXpoints = [pointsX[q] for q in excRepsIndex]
        excYpoints = [pointsY[q] for q in excRepsIndex]
        excSize    = [diameter for q in excRepsIndex]

        #if len(excRepsIndex) != 0:
        #    print(excRepsIndex)
        #    print(excXpoints)
        #    print(excYpoints)
        #    print(excSize)

        ax[s1//3][s1%3].scatter(pointsX, pointsY, s=sizesTotal, facecolors='none', edgecolors='r')
        ax[s1//3][s1%3].scatter(pointsX, pointsY, s=sizesEmp, facecolors='b', edgecolors='b')
        #ax[s1//3][s1%3].scatter(excXpoints, excYpoints, s=excSize, marker='+', facecolor='y')
        #ax[s1 // 3][s1 % 3].scatter([0], [0], s=sizesEmp, facecolors='g', edgecolors='g')

        identf = plt.imread("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/vizualizer/images/{}.png".format(s1+1))
        imscatter(-40, -30, identf, ax[s1//3][s1%3])
        ax[s1 // 3][s1 % 3].set_title(titles[s1], x=-0.1, fontname="Times New Roman", fontweight="bold")

    plt.savefig("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/vizualizer/gainNeurons/2-1vpc/{}_neuronGainMod.png".format(neuron))
    #plt.show()

def gainModulations(activations, emActivations):
    print("activation", activations.shape)
    results = []
    for i in range(activations.shape[0]):
        modulatedData = []
        for j in range(activations.shape[1]):
            totalResponse = activations[i, j]
            diffResponse =  totalResponse - emActivations[i, j]

            modulatedData.append((totalResponse, diffResponse))
        results.append(modulatedData)
    results = np.array(results)
    print(results.shape)
    return results

def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

if __name__ == '__main__':
    imagespath = '/home/martin/data/v2/gain_mod_testing/'

    ### for neuron in range(15):
    ###     activations, emActivations = testGainFields(savepath, imagespath, neuron=neuron)
    ###     results = gainModulations(activations, emActivations)
    ###     plotGainModulations(results, neuron)
    for neuron in range(0, 28):
        activations, emActivations = testGainFields(imagespath, neuron=neuron, version="asdasd")
        results = gainModulations(activations, emActivations)
        plotGainModulations(results, neuron)

