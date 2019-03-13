from training.FRMW_UBAL.datapreparation.plotDataset import extractFeatures
from training.FRMW_UBAL.UbalNet import loadModel
from sklearn.externals import joblib
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib import patches

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

def testGainFields(pathToSave, imagespath, way="v", neuron=1):
    retinalRedInf = getRetinalReducedDataToTest(imagespath)
    tilts = [9.8, 1, -10, -19]
    versions = [29, 10, 0, -10, -19]
    vergence = 30

    loadScaler = joblib.load("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/scaler/sc3.pkl")
    model = loadModel('/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/ret2/')
    idx = 1
    path = pathToSave + '/' if pathToSave[-1] != '/' else pathToSave
    resultActs = list()
    emResultActs = list()
    #resultSums = list()

    for titlIdx in range(len(tilts)):
        for versionIdx in range(len(versions)):
            activationForTitlAndVers = []
            emActivationForTitlAndVers = []
            #sumForTitlAndVers = []
            for i in range(len(retinalRedInf)):
                inputNonScaled = [0] * 16
                inputNonScaled[0] = tilts[titlIdx]
                inputNonScaled[1] = versions[versionIdx]
                inputNonScaled[2] = vergence

                for j in range(3):
                    inputNonScaled[3+j] = retinalRedInf[i][j]
                for k in range(3):
                    inputNonScaled[6+k] = retinalRedInf[i][k+3]

                #Scale created input with the scaler used for testing data
                scaledInput = loadScaler.transform([inputNonScaled])
                x = scaledInput[0][:9]

                # Run forward predition on the best model
                model.predictForward(x)
                hiddenActivations = model.get_last_activations()[neuron]
                #hiddenSums = model.get_last_sums()


                x[0] = 0
                x[1] = 0
                x[2] = 0
                model.predictForward(x)
                emphiddenActivations = model.get_last_activations()[neuron]
                #emphiddenSums = model.get_last_sums()

                activationForTitlAndVers.append(hiddenActivations)
                emActivationForTitlAndVers.append(emphiddenActivations)
                #sumForTitlAndVers.append(hiddenSums)

            resultActs.append(activationForTitlAndVers)
            emResultActs.append(emActivationForTitlAndVers)
            #resultSums.append(sumForTitlAndVers)

    return np.array(resultActs), np.array(emResultActs) #, np.array(resultSums)


def plotGainModulations(results, neuron):
    idx = 0
    dirm = .09
    pos1 = 0.17
    pos2 = 0.25
    mrg = .05
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))

    minI = 0
    maxI = 19
    maxA, maxE = max(results[minI:maxI, 0]), max(results[minI:maxI, 1])
    print(maxE, maxA)

    for k0 in range(3):
        for k1 in range(3):
            patchesArr = []
            for i in range(4):
                for j in range(5):
                    spaceI = dirm if i != 0 else 0
                    circ1 = patches.Circle((.12 + ((dirm*i*2)+0.07*i), .9-((dirm*j*2))-0.02*j), dirm*(results[idx, 0]/maxA), alpha=0.8, fc='blue')
                    circ2 = patches.Circle((.12 + ((dirm*i*2)+0.07*i), .9-((dirm*j*2))-0.02*j), dirm*(results[idx, 1]/maxE), alpha=0.8, fc='red')
                    patchesArr.append(circ1)
                    patchesArr.append(circ2)
                    idx += 1
                    print(results[idx, 0], results[idx, 1])

            collection = PatchCollection(patchesArr, match_original=True)
            ax[k0][k1].add_collection(collection)
            minI = maxI
            maxI = maxI+19
            maxA, maxE = max(results[minI:maxI, 0]), max(results[minI:maxI, 1])
            #  print(minI, maxI)
            #ax[k0][k1].axis("off")

    plt.subplots_adjust(left=mrg, right=1-mrg, bottom=mrg, top=1-mrg)
    fig.savefig("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/vizualizer/gainNeurons/"+str(neuron)+".png")

def gainModulations(activations, emActivations):
    results = []
    for i in range(activations.shape[0]):
        for j in range(activations.shape[1]):
            totalResponse = activations[i, j]
            diffResponse =  activations[i, j] - emActivations[i, j]
            results.append((totalResponse, diffResponse))

    return np.array(results)


if __name__ == '__main__':
    savepath = '/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/vizualizer/data/'
    imagespath = '/home/martin/data/v2/gain_mod_testing/'

    for neuron in range(1):
        activations, emActivations = testGainFields(savepath, imagespath, neuron=neuron)
        results = gainModulations(activations, emActivations)
        plotGainModulations(results, neuron)

