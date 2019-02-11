import numpy as np
import time

start_time = time.time()

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def XOR():
    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    y = np.array([0, 1, 1, 0]).reshape(4, 1)
    return X, y


def AND():
    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    y = np.array([1, 0, 0, 0]).reshape(4, 1)
    return X, y


def random_patterns():
    X = np.array([[1, 0, 0, 0, 0, 1, 0, 1],
                  [0, 1, 0, 0, 0, 0, 1, 1],
                  [0, 0, 1, 0, 1, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0]])
    y = np.array([[1, 0, 0, 0, 1, 1],
                  [0, 1, 0, 0, 1, 0],
                  [0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 1, 1]])
    return X, y

# data
X, y = random_patterns()
#X, y = XOR()

# net architecture
numInp = 8  # number of features in dataset
numHid = 6
numOut = 6 #6

# hyperparameters
lr = 0.1
epochs = 5000

f_beta_input = 1
f_beta_hidden = 0.5
f_beta_output = 0

b_beta_input = 1 - f_beta_input
b_beta_hidden = 1 - f_beta_hidden
b_beta_output = 1 - f_beta_output

gamma_1 = 0
gamma_2 = 0
gamma_3 = 0
gamma_4 = 0.5

# initialize weights
w1 = np.random.uniform(size=(numInp, numHid))
b1 = np.random.uniform(size=(1, numHid))
w2 = np.random.uniform(size=(numHid, numOut))
b2 = np.random.uniform(size=(1, numOut))

m1 = np.random.uniform(size=(numOut, numHid))
d1 = np.random.uniform(size=(1, numHid))
m2 = np.random.uniform(size=(numHid, numInp))
d2 = np.random.uniform(size=(1, numInp))

for epoch in range(epochs):
    f_sum_error, b_sum_error = 0, 0
    f_sum_ratio, b_sum_ratio = 0, 0
    # forward phase
    qhfp = sigmoid(np.dot(X, w1) + b1)  # hidden prediction         ### Forward prediction
    pofe = sigmoid(np.dot(qhfp, m2) + d2)  # output echo            ### Forward echo

    qofp = sigmoid(np.dot(qhfp, w2) + b2)  # output prediction      ###
    phfe = sigmoid(np.dot(qofp, m1) + d1)  # hidden echo

    # backward phase
    phbp = sigmoid(np.dot(y, m1) + d1)  # hidden prediction
    qobe = sigmoid(np.dot(phbp, w2) + b2)  # output echo

    pobp = sigmoid(np.dot(phbp, m2) + d2)  # output prediction
    qhbe = sigmoid(np.dot(pobp, w1) + b1)  # hidden echo

    # learning rules hidden
    tfh = f_beta_hidden * qhfp + (1-b_beta_hidden) * phbp
    efh = gamma_4 * qhfp + (1-gamma_4) * qhbe
    tbh = b_beta_hidden * phbp + (1-f_beta_hidden) * qhfp
    ebh = gamma_4 * phbp + (1-gamma_4) * phfe

    # learning rules output
    tfo = f_beta_output * qofp + (1-f_beta_output) * y
    efo = gamma_4 * qofp + (1-gamma_4) * qobe
    tbo = b_beta_input * pobp + (1-b_beta_input) * X
    ebo = gamma_4 * pobp + (1-gamma_4) * pofe

    # update weights and biases
    delta_h_f = lr * np.dot(tbo.T, (tfh - efh))
    w1 += delta_h_f
    b1 += np.sum(tfh - efh, axis=0)

    delta_o_f = lr * np.dot(tbh.T, (tfo - efo))
    w2 += delta_o_f
    b2 += np.sum(tfo - efo, axis=0)

    delta_h_b = lr * np.dot(tfo.T, (tbh - ebh))
    m1 += delta_h_b
    d1 += np.sum(tbh - ebh, axis=0)

    delta_o_b = lr * np.dot(tfh.T, (tbo - ebo))
    m2 += delta_o_b
    d2 += np.sum(tbo - ebo, axis=0)

    # sum_error
    f_error = (np.where(qofp > .5, 1, 0) - y)**2
    b_error = (np.where(pobp > .5, 1, 0) - X)**2
    f_sum_error += np.sum(f_error)
    b_sum_error += np.sum(b_error)

    f_sum_ratio += np.sum(np.sqrt((qofp - y)**2))
    b_sum_ratio += np.sum(np.sqrt((pobp - X)**2))

    print('epoch: {}   '
          'forward error: {}   '
          'backward error: {}   '
          'forward loss: {}   '
          'backward loss: {}   '.format(epoch, f_sum_error, b_sum_error, f_sum_ratio, b_sum_ratio))

    if f_sum_error == 0 and b_sum_error == 0:
        break

print("--- %.3gs seconds ---" % (time.time() - start_time))
