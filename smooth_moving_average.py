import numpy
import numpy.core
import math

def smooth1D(x,window_len,move_to_middle,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise Exception("smooth only accepts 1 dimension arrays.")

    if len(x) < window_len:
        raise Exception("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise Exception("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    
    if move_to_middle:
        z = numpy.zeros(len(x))
        for i in range(len(x)):
            z[i] = y[i+math.floor((window_len-1)/2)]
        y = z
        
    return y

def smooth(x, window_size, move_to_middle, window='hanning'):
    x_array = numpy.asarray(x)
    output = []
    for i in range(0, x_array.shape[1]):
        s = smooth1D(x_array[:,i], window_size, move_to_middle, window)
        output.append(s)
    output = numpy.asarray(output)
    # Transpose the matrix
    output = output.T.tolist()
    return output
