from training.FRMW_UBAL.UbalNet import loadModel
from sklearn.externals import joblib
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import math


def findSuitableData(path, neuron):
    sizeOfObejct = 0
    toHeatMap = list()
    model = loadModel(path)
    loadScaler = joblib.load("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/scaler/sc3.pkl")
    hiddenWeights = model.W1[3:, neuron]
    maxValue = 0

    for h in range(0, 240):
        row = list()
        for w in range(0, 320):
            inputNonScaled = [0] * 16
            inputNonScaled[3] = w
            inputNonScaled[6] = w
            inputNonScaled[4] = h
            inputNonScaled[7] = h
            inputNonScaled[5] = sizeOfObejct
            inputNonScaled[8] = sizeOfObejct

            scaledInput = loadScaler.transform([inputNonScaled])[0]
            scaledInput[5] = 0
            scaledInput[8] = 0

            sumV = np.dot(scaledInput[3:9], hiddenWeights)
            row.append(sumV)
            maxValue = maxValue if maxValue > sumV else sumV

        toHeatMap.append(row)

    xticks = np.arange(20, 320, 20)
    yticks = np.arange(20, 240, 20)

    plt.figure()
    ax = sns.heatmap(toHeatMap, cmap="Reds", xticklabels=xticks, yticklabels=yticks)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    plt.savefig("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/vizualizer/rcPlots/RC_neuron{}.png".format(neuron))

if __name__ == '__main__':
    for neuron in range(15):
        findSuitableData("/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/models/ret2/", neuron=neuron)