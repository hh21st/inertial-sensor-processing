import numpy
from scipy import signal

def smooth(x,window_size):
    x_array = numpy.asarray(x)
    output = []
    for i in range(0, x_array.shape[1]):
        s = signal.medfilt(x_array[:,i], window_size)
        output.append(s)
    output = numpy.asarray(output)
    # Transpose the matrix
    output = output.T.tolist()
    return output
