import matplotlib.pyplot as plt
import numpy as np

data = {}
data["SGD"] = np.load("SGD.npy")
data["Adam"] = np.load("Adam.npy")
data["AdamW"] = np.load("AdamW.npy")
data["RMSProp"] = np.load("RMSProp.npy")

for k in data.keys():
    plt.plot(data[k], label = k)

plt.ylabel('error')
plt.xlabel('steps')
plt.legend()
plt.show()