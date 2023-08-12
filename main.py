from preprocess import *
from tensorflow.keras.models import load_model

def inference(filename):
    data = np.load(filename)
    dataset = loads(data)
    wavelet = "mexh"  # mexh, morl, gaus8, gaus4
    scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 101, 1)

    x1, x2 = worker(dataset, wavelet=wavelet, scales=scales, sampling_period=1. / sampling_rate)
    x1_test = np.reshape(x1, (len(x1), 100, 100, 1))

    model=load_model('wavelet.h5')

    y = model.predict(x1_test)

    y = y.round()

    y_label = np.argmax(y, axis=1)

    # map to class
    N, S, V, F, Q = 0, 0, 0, 0, 0
    for i in y_label:
        if i == 0:
            N += 1
        elif i == 1:
            S += 1
        elif i == 2:
            V += 1
        elif i == 3:
            F += 1
        elif i == 4:
            Q += 1

    # print result 
    print("N: ", N)
    print("S: ", S)
    print("V: ", V)
    print("F: ", F)
    print("Q: ", Q)
    
    return N, S, V, F, Q