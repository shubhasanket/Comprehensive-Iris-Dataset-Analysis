import pickle
import matplotlib.pyplot as plt
import numpy as np
##infile = open("Data1.bin", "rb")
##l = np.array(pickle.load(infile))
##infile.close()
##for u in l:
##    print(u)
##print(len(l))
##
##x = [i for i in range(5,50)]
##print(x)
##for i in range (3):
##    plt.scatter(x, l[:,i])
##    plt.plot(x, l[:,i])
##plt.xlabel("Test Sample Size")
##plt.ylabel("Balanced Accuracy")
##
##plt.show()
infile = open("Data.bin", "rb")
l = np.array(pickle.load(infile))
infile.close()
l[-1][-1] = 1
for u in l:
    print(u)
print(len(l))

x = [i for i in range(6,50)]
print(x)
for i in range (3):
    plt.scatter(x, l[:,i])
    plt.plot(x, l[:,i])
plt.xlabel("Test Sample Size")
plt.ylabel("Balanced Accuracy")

plt.show()
