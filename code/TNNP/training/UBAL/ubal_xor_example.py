import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def XOR():
    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    y = np.array([0, 1, 1, 0]).reshape(4, 1)
    return X, y


# data
X, y = XOR()

# net architecture
numInp = 2  # number of features in dataset
numHid = 10
numOut = 1

# hyperparameters
lr = 2
epochs = 5000
beta_fwd = [0, 1, 0]
beta_bwd = [1, 0, 1]
gamma_fwd = [0.5, 0.5]
gamma_bwd = [0.5, 0.5]

count = 0
converge_list = []
for i in range(1):
    np.random.seed(i)
    print(i)
    m, sigma = 0, 2.5
    w1 = np.random.normal(loc=m, scale=sigma, size=(numInp, numHid))
    b1 = np.random.normal(loc=m, scale=sigma, size=(1, numHid))
    w2 = np.random.normal(loc=m, scale=sigma, size=(numHid, numOut))
    b2 = np.random.normal(loc=m, scale=sigma, size=(1, numOut))

    m1 = np.random.normal(loc=m, scale=sigma, size=(numOut, numHid))
    d1 = np.random.normal(loc=m, scale=sigma, size=(1, numHid))
    m2 = np.random.normal(loc=m, scale=sigma, size=(numHid, numInp))
    d2 = np.random.normal(loc=m, scale=sigma, size=(1, numInp))
    for epoch in range(epochs):
        f_sum_error, b_sum_error = 0, 0
        f_sum_ratio, b_sum_ratio = 0, 0
        import random
        indices = [0, 1, 2, 3]
        random.shuffle(indices)
        for example in indices:  # for stochastic learning
            # fwd phase
            qhfp = sigmoid(np.dot(X[example], w1) + b1)  # hidden prediction
            pofe = sigmoid(np.dot(qhfp, m2) + d2)  # output echo

            qofp = sigmoid(np.dot(qhfp, w2) + b2)  # output prediction
            phfe = sigmoid(np.dot(qofp, m1) + d1)  # hidden echo

            # bwd phase
            phbp = sigmoid(np.dot(y[example], m1) + d1)  # hidden prediction
            qobe = sigmoid(np.dot(phbp, w2) + b2)  # output echo

            pobp = sigmoid(np.dot(phbp, m2) + d2)  # output prediction
            qhbe = sigmoid(np.dot(pobp, w1) + b1)  # hidden echo

            # learning rules fwd
            tfh = beta_fwd[1] * qhfp + (1 - beta_fwd[1]) * phbp
            efh = gamma_fwd[0] * qhfp + (1 - gamma_fwd[0]) * qhbe
            tfo = beta_fwd[2] * qofp + (1 - beta_fwd[2]) * y[example]
            efo = gamma_fwd[0] * qofp + (1 - gamma_fwd[0]) * qobe

            # learning rules bwd
            tbh = beta_bwd[1] * phbp + (1 - beta_bwd[1]) * qhfp
            ebh = gamma_bwd[0] * phbp + (1 - gamma_bwd[0]) * phfe
            tbo = beta_bwd[0] * pobp + (1 - beta_bwd[0]) * X[example]
            ebo = gamma_bwd[1] * pobp + (1 - gamma_bwd[1]) * pofe

            # update weights and biases
            delta_h_f = lr * np.outer(tbo, (tfh - efh))
            w1 += delta_h_f
            b1 += (tfh - efh)

            delta_o_f = lr * np.outer(tbh, (tfo - efo))
            w2 += delta_o_f
            b2 += (tfo - efo)

            delta_h_b = lr * np.outer(tfo, (tbh - ebh))
            m1 += delta_h_b
            d1 += (tbh - ebh)

            delta_o_b = lr * np.outer(tfh, (tbo - ebo))
            m2 += delta_o_b
            d2 += (tbo - ebo)

            # sum_error
            f_error = (np.where(qofp > .5, 1, 0) - y[example])**2
            b_error = (np.where(pobp > .5, 1, 0) - X[example])**2
            f_sum_error += np.sum(f_error)
            b_sum_error += np.sum(b_error)

            f_sum_ratio += np.sum(np.sqrt((qofp - y[example])**2))
            b_sum_ratio += np.sum(np.sqrt((pobp - X[example])**2))

        if epoch%10 == 0:
       	    print('epoch: {}   '
           	    'fwd error: {}   '
                'bwd error: {}   '
                'fwd loss: {}   '
                'bwd loss: {}   '.format(epoch, f_sum_error, b_sum_error, f_sum_ratio, b_sum_ratio))

        if f_sum_error == 0:
            converge_list.append(i)
            count += 1
            break

print("nets converged:", count)
print("converged_list:", converge_list)
