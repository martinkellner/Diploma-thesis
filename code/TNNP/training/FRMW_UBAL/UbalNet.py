import numpy as np
import matplotlib.pyplot as plt
from training.FRMW_UBAL.DataPreparation import getData

class UbalNet:

    def __init__(self):
        self.W1 = None
        self.W2 = None
        self.M1 = None
        self.M2 = None
        self.b1 = None
        self.b2 = None
        self.d1 = None
        self.d2 = None

        self.af = ""
        self.alpha = None
        self.epochs = None
        self.numHidden = None
        self.f_beta = [1, .5, 0]
        self.b_beta = [0, .5, 1]
        self.gamma = [0, 0, 0, 0.5]
        self.count = None

        self.fig, self.axs = plt.subplots(2, 1)

    def forward_prediction(self, input, weights, bias):
        return self.activation(np.dot(input, weights) + bias)

    def forward_echo(self, prediction, weights, bias):
        return self.activation(np.dot(prediction, weights) + bias)

    def backward_predition(self, output, weights, bias):
        return self.activation(np.dot(output, weights) + bias)

    def backward_echo(self, prediction, weights, bias):
        return self.activation(np.dot(prediction, weights) + bias)

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("Wrong dimensions, numbers of X and y must be the same!")

        self.count = X.shape[0]
        self.W1 = np.random.uniform(size=(X.shape[1], self.numHidden))
        self.b1 = np.random.uniform(size=(1, self.numHidden))
        self.W2 = np.random.uniform(size=(self.numHidden, y.shape[1]))
        self.b2 = np.random.uniform(1, y.shape[1])

        self.M1 = np.random.uniform(size=(y.shape[1], self.numHidden))
        self.d1 = np.random.uniform(size=(1, self.numHidden))
        self.M2 = np.random.uniform(size=(self.numHidden, X.shape[1]))
        self.d2 = np.random.uniform(size=(1, X.shape[1]))

        self.training(X, y)

    def saveWeights(self):
        """TODO: implement saving weight matrixes into a file"""
        pass

    def predict(self, X, y):
        pass

    def training(self, X, y):
        fError = []
        bError = []
        indexs = np.arange(self.count)

        for ep in range(self.epochs):
            epFError = []
            epBError = []
            np.random.shuffle(indexs)

            for idx in indexs:
                qhfp = self.forward_prediction(X[idx], self.W1, self.b1)
                pofe = self.forward_echo(qhfp, self.M2, self.d2)

                qofp = self.forward_prediction(qhfp, self.W2, self.b2)
                phfe = self.forward_echo(qofp, self.M1, self.d1)

                phbp = self.backward_predition(y[idx], self.M1, self.d1)
                qobe = self.backward_echo(phbp, self.W2, self.b2)

                pobp = self.backward_predition(phbp, self.M2, self.d2)
                qhbe = self.backward_echo(pobp, self.W1, self.b1)

                tfh = self.f_beta[1] * qhfp + (1 - self.b_beta[1]) * phbp
                efh = self.gamma[3] * qhfp + (1 - self.gamma[3]) * qhbe
                tbh = self.b_beta[1] * phbp + (1 - self.f_beta[1]) * qhfp
                ebh = self.gamma[3] * phbp + (1 - self.gamma[3]) * phfe

                tfo = self.f_beta[2] * qofp + (1 - self.f_beta[2]) * y[idx]
                efo = self.gamma[3] * qofp + (1 - self.gamma[3]) * qobe
                tbo = self.b_beta[0] * pobp + (1 - self.b_beta[0]) * X[idx]
                ebo = self.gamma[3] * pobp + (1 - self.gamma[3]) * pofe

                # update weights and biases
                delta_h_f = self.alpha * np.dot(tbo.T, (tfh - efh))
                self.W1 += delta_h_f
                self.b1 += np.sum(tfh - efh, axis=0)

                delta_o_f = self.alpha * np.dot(tbh.T, (tfo - efo))
                self.W2 += delta_o_f
                self.b2 += np.sum(tfo - efo, axis=0)

                delta_h_b = self.alpha * np.dot(tfo.T, (tbh - ebh))
                self.M1 += delta_h_b
                self.d1 += np.sum(tbh - ebh, axis=0)

                delta_o_b = self.alpha * np.dot(tfh.T, (tbo - ebo))
                self.M2 += delta_o_b
                self.d2 += np.sum(tbo - ebo, axis=0)

                epFError.append(self.lostFuntion(y[idx], qofp))
                epBError.append(self.lostFuntion(X[idx], pobp))

            fError.append(np.mean(epFError))
            bError.append(np.mean(epBError))

            print("Epoch: {}\tForward MSE:{}\tBackward MSE:{}".format(ep, epFError[-1], epBError[-1]))

        self.plotError(fError, bError, ep)

    def plotError(self, errF, errB, ep):
        print("Epoch: {}    F_MSE: {},     B_MSE: {}".format(ep, errF[-1], errB[-1]))

        self.axs[0].plot(np.arange(len(errF)), errF)
        self.axs[0].set_title("Forward MSE error")
        self.axs[0].set_xlabel("Epochs")
        self.axs[0].set_ylabel("FP MSE")

        #self.fig.suptitle("Errors")

        self.axs[1].plot(np.arange(len(errB)), errB)
        self.axs[1].set_title("Backward MSE error")
        self.axs[1].set_xlabel("Epochs")
        self.axs[1].set_ylabel("BP MSE")

        plt.show()

    def lostFuntion(self, targets, outputs):
        lost = np.sum((targets - outputs) ** 2)
        #print(lost)
        return lost

    def activation(self, x):
        if self.af == "sigmoid":
            return 1 / (1 + np.exp(-x))

    def setHyperparamenters(self, alpha=0.1, epochs=100, numhidden=10, activationn="sigmoid"):
        self.alpha = alpha
        self.epochs = epochs
        self.numHidden = numhidden
        self.af = activationn


if __name__ == '__main__':
    X, y = getData()
    un1 = UbalNet()
    un1.setHyperparamenters(.05, 30, 15)
    un1.fit(X, y)