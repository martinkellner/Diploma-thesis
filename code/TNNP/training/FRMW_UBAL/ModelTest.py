from training.FRMW_UBAL.UbalNet import loadModel
from training.FRMW_UBAL.featureExtractor import extractFeatures
import sys
from sklearn.externals import joblib
import numpy as np

if __name__ == '__main__':
    model1 = "/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/1-1v2/"
    model1Scaler = "/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/1-1v2/scaler.pkl"
    saveRes = "/home/martin/School/Diploma-thesis/code/channel.txt"
    model2A = '/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/results/2-1v1/'
    model2B = None
    inputarr = [float(sys.argv[i]) for i in range(2, len(sys.argv))]
    result = None
    forward = True if len(inputarr) == 3 else False
    imgl = '/home/martin/School/Diploma-thesis/code/l_img.ppm'
    imgr = '/home/martin/School/Diploma-thesis/code/r_img.ppm'

    idf = sys.argv[1]
    if idf == "1":
        model = loadModel(model1)
        scaler = joblib.load(model1Scaler)

        inputarr = inputarr + [0]*7 if forward else [0]*3 + inputarr
        inputarr = np.array([inputarr])
        inputarrSlc = scaler.transform(inputarr)

        inputNet = inputarrSlc[0][0:3] if forward else inputarrSlc[0][3:]
        prediction = [[0]*3 + model.predictForward(inputNet)[0].tolist()] if forward else [model.predictBackward(inputNet)[0].tolist() + [0]*7]
        unScaledPred = scaler.inverse_transform(prediction)
        result = unScaledPred[0][3:] if forward else unScaledPred[0][0:3]
        savetxt = ""
        for idx in range(len(result)):
            savetxt += str(result[idx])
            if idx == len(result)-1:
                savetxt += '\n'
            else:
                savetxt += ' '
        with open(saveRes, 'w+') as file:
            print(savetxt, file=file)

    result = None
    if idf == '2':
        model = loadModel(model2A)
        scaler = joblib.load('/home/martin/School/Diploma-thesis/code/TNNP/training/FRMW_UBAL/scaler/sc_2-1.pkl')
        if forward:
            xl, yl, sl = extractFeatures(imgl)
            xr, yr, sr = extractFeatures(imgr)
            #print(xl, yl, sl, xr, yr, sr)
            inputarr = inputarr + [xl, yl, sl] + [xr, yr, sr] + [0]*7
            inputarr = np.array([inputarr])
            inputarrSlc = scaler.transform(inputarr)
            #print(inputarrSlc)
            inputNet = np.array([inputarrSlc[0][:9]])
            print(inputNet)
            prediction = [[0] * 9 + model.predictForward(inputNet)[0].tolist()]
            unScaledPred = scaler.inverse_transform(prediction)
            result = unScaledPred[0][9:]
        else:
            inputarr = np.array([[0]*9 + inputarr])
            inputarrSlc = scaler.transform(inputarr)
            inputNet = np.array([inputarrSlc[0][9:]])
            print(inputNet)
            prediction = [model.predictBackward(inputNet)[0].tolist() + [0]*7]
            unScaledPred = scaler.inverse_transform(prediction)
            result = unScaledPred[0][:3]

        savetxt = ""
        for idx in range(len(result)):
            savetxt += str(result[idx])
            if idx == len(result) - 1:
                savetxt += '\n'
            else:
                savetxt += ' '
        with open(saveRes, 'w+') as file:
            print(savetxt, file=file)