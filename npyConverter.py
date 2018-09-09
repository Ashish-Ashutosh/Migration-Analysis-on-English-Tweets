import numpy as np
import numpy
#numpy.set_printoptions(threshold=numpy.nan)
#csv = np.genfromtxt('Negative.csv', delimiter=",")
#np.savetxt("Negative.csv", "negative.npy", delimiter=";")

data = np.genfromtxt('Positive_appended.csv', skip_header=True, delimiter=";", dtype=None)
np.save('positive.npy',data)
