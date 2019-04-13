import numpy as np
import operator
import matplotlib.pyplot as plt

### Inspired from Svec 20xx (TODO: replace xx with the correct value)
def uniformPeakPlacement(minValue, maxValue, numberOfNeurons):
    step = (maxValue-minValue)/(numberOfNeurons-1)
    return list(minValue+(idx*step) for idx in range(numberOfNeurons))

### Inspired from Svec 20xx (TODO: replace xx with the correct value)
def gaussian(x, h, w, p):
    return h*np.exp( -1*( ((x-p)**2) / (2*(w**2) )))

### Inspired from Svec 20xx (TODO: replace xx with the correct value)
def invGaussian(y, h, w, p):
    det = (-2*p)**2 - 4*(p**2 + 2*(w**2)*np.log(y/h))
    return (2*p - np.sqrt(det))/ 2, (2*p + np.sqrt(det))/ 2

### Inspired from Svec 20xx (TODO: replace xx with the correct value)
def gaussianPopulationCoding(value, peaks, h, w):
    return list(gaussian(value, h, w, peaks[i]) for i in  range(len(peaks)))

### Inspired from Svec 20xx (TODO: replace xx with the correct value)
def gaussianPopulationDecoding(y, peaks, h, w):
    res = None
    maxValue = -1
    maxPeakIdx = -1
    maxPeakIdx, maxValue = max(enumerate(y), key=operator.itemgetter(1))
    x1, x2 = invGaussian(maxValue, h, w, peaks[maxPeakIdx])
    if maxPeakIdx == 0 or (maxPeakIdx<len(peaks)-1 and y[maxPeakIdx+1] > y[maxPeakIdx-1]):
        res = x2
    else:
        res = x1
    if maxPeakIdx>0:
        _, x2 = invGaussian(y[maxPeakIdx-1], h, w, peaks[maxPeakIdx-1])
        res = (x2+res)/2
    if maxPeakIdx<len(peaks)-1:
        x1, _ = invGaussian(y[maxPeakIdx+1], h, w, peaks[maxPeakIdx+1])
        res = (x1 + res) / 2

    return res

def plotImage(interval, peaks, w, h, encoded, real):


    for p in range(len(peaks)):
        x = []
        y = []

        for i in interval:
            y.append(gaussian(i, h, w, peaks[p]))
            x.append(i)

        rp = plt.plot(x, y)
        if encoded[p] > 0:
            plt.plot(real, encoded[p], marker='o', markersize=10, color=rp[0].get_color())

        plt.xlabel("Direction of movement (degrees)")
        plt.ylabel("Activity (spikes/sec)")

if __name__ == '__main__':
    #Testing

    #enpop = gaussianPopulationCoding(100, peaks, 1, 10)
    #print(gaussianPopulationDecoding(enpop, peaks, 1, 10))

    # image
    neurons = 5
    w = 7
    real = 12

    peaks = uniformPeakPlacement(-30, 30, neurons)
    print(peaks)
    encoded = gaussianPopulationCoding(real, peaks, 1, w)
    print(encoded)
    plotImage(np.arange(-30, 30, 0.001), peaks, w, 1, encoded, real)
