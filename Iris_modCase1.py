from sklearn import datasets
import matplotlib.pyplot as plt
import random
import numpy as np
import math
from tabulate import tabulate
import pickle
import sys

DEBUG = not True
DEBUG2 = not True
DEBUG3 = not True

class Iris:
    def __init__(self):
        '''
        setosa = 0
        versicolor = 1
        virginica = 2
        '''
        iris = datasets.load_iris()
        self.classes = 3
        self.features = 4
##        print(iris)
        self.data_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
        self.target_names = iris['target_names']
        if DEBUG: print(self.target_names)
        self.iris_x = iris['data']
        if DEBUG: print(self.iris_x)
        self.iris_y = iris['target']
        if DEBUG: print(self.iris_y)

        # Creating a dictionary of the form:
        #{0: data in list form, 1: data in list form, ...
        self.data = {0: [], 1: [], 2: []}
        i = 0
        for u in self.iris_y:
            self.data[u].append(list(self.iris_x[i]))
            i += 1
        
        # Every class has 50 data points.
        self.classSize = 50
        # In model 1 we will pick 40 points from each class to train our model
        # for each sample size, generating a random model 15 times by randomizing the training sample
        # and determining its balanced accuracy
        self.sigma_case = 3
        self.final_models = [np.array([0]*self.features), np.array([0]*self.features), np.array([0]*self.features),
                             np.array([[0]*self.features]*self.features), np.array([[0]*self.features]*self.features),
                             np.array([[0]*self.features]*self.features)]
        print(self.final_models)
        self.additions = 0
        data = []
        num_of_times = 30
        for i in range (6,self.classSize):
            u = self.repete(i,num_of_times)
            data.append(u)
            print(u)
        print(data)
        out_file = open("Data.bin", "wb")
        pickle.dump(data,out_file)
        self.MLE_estimate(30)
        print(self.final_models)
        print(self.additions)
        for i in range (len(self.final_models)):
            self.final_models[i] = np.multiply(1/self.additions, self.final_models[i])
        for i in range (self.classes):
            self.mean_estimate[i] = self.final_models[i]
            self.sigma_estimate[i] = self.final_models[i+self.classes]

        self.generate_sigma_inv()
        print(self.final_test())  
        

    def repete(self,num_of_training_data,num_of_times):
        # l = [min, max, average]
        add = False
        l = [100,0,0]
        for i in range (num_of_times):
            a = self.MLE_estimate(num_of_training_data)
            if a < l[0]:
                l[0] = a
            if a > l[1]:
                l[1] = a
            l[-1] += a
            try:
                if (int(a) == 1 and (not add) and (20 <= num_of_training_data <= 30)):
                    for i in range (self.classes):
                        self.final_models[i] = np.add(self.final_models[i],self.mean_estimate[i])
                        self.final_models[i+self.classes] = np.add(self.final_models[i+self.classes],self.sigma_estimate[i])
                    add = True
                    self.additions += 1
            except:
                continue
            
        l[-1] = l[-1]/num_of_times
        return l
        
    def MLE_estimate(self,num_of_training_data):
        '''Estimates the mean and the covariance matrix using MLE
            Here, self.mean_estimate, self.sigma_estimate, and
            self.sigma_inverse are being defined
        '''
        # Randomizing the sample we have
        for _ in range (self.classes):
            random.shuffle(self.data[_])

        self.num_of_data_points_used_for_training = num_of_training_data

        # For a gaussian, the estimate of the mean using MLE is known
        # Using this we compute the mean and the covariance matrix of each of the classes.

        self.mean_estimate = {0: [], 1: [], 2: []}
        for _ in range (self.classes):
            matrix = np.array([0]*self.features)
            for i in range (self.num_of_data_points_used_for_training):
                if DEBUG2:
                    print(matrix, self.data[_][i])
                matrix = np.add(matrix, np.array(self.data[_][i]))
            if DEBUG2:
                print(matrix)
            self.mean_estimate[_] = np.multiply(np.array([1/self.num_of_data_points_used_for_training]),matrix)
##        print(self.mean_estimate)
        # The mean is ready

        self.sigma_estimate = {0: [], 1: [], 2: []}
        for _ in range (self.classes):
            if DEBUG3:  
                print("Estimate for Class " + str(_))
                print("##########################################################")
                print("##########################################################")
            matrix = np.array([[0]*self.features]*self.features)
            for i in range (self.num_of_data_points_used_for_training):
                temp = np.subtract(np.array(self.data[_][i]), self.mean_estimate[_])
                if DEBUG3:
                    print("data:", np.array(self.data[_][i]))
                    print("mean estimate:", self.mean_estimate[_])
                    print("temp:", temp)
                temp = np.outer(temp,temp)
                if DEBUG3:
                    print("matrix:",matrix)
                    print("temp:",temp)
                matrix = np.add(matrix, temp)
##            self.sigma_estimate[_] = np.multiply(np.array([1/self.num_of_data_points_used_for_training]),matrix)
                self.sigma_estimate[_] = np.multiply(1/self.num_of_data_points_used_for_training,matrix)
##            print(matrix)
##        print(self.sigma_estimate)
##        for u in self.sigma_estimate:
##            print(self.sigma_estimate[u])
        # The covariance matrix is ready
        self.sigma_inverse = {}
        for _ in range(self.classes):
            try:
                self.sigma_inverse[_] = np.linalg.inv(self.sigma_estimate[_])
            except:
##                a = np.linalg.eig(self.sigma_estimate[_])
##                val = a[0]
##                vec = a[1].T
##                m = np.array([[0]*len(val)]*len(val))
##                for i in range(len(val)):
##                    if val[i] != 0:
##                        
##                        m += np.multiply([1/val[i]],np.outer(vec[i], vec[i]))
##                print(m)

                
                for i in range(self.features):
                    if abs(self.sigma_estimate[_][i][i]) == 0:
                        self.sigma_estimate[_][i][i] = 10**(-6)
##                    print(self.sigma_estimate[_])
                self.sigma_inverse[_] = np.linalg.inv(self.sigma_estimate[_])



                
##                a = np.linalg.svd(self.sigma_estimate[_])
##                print(a)
##                U = a[0]
##                D = a[1]
##                VT = a[2]
##                nD = []
##                for i in range(len(D)):
##                    nD.append([])
##                    for j in range(len(D)):
##                        if j != i:
##                            nD[-1].append(0)
##                        else:
##                            if abs(D[i]) != 0:
##    ##                        print(D[i])
##                                nD[-1].append(1/D[i])
##                            else:
##                                nD[-1].append(0)
##                print(U)
##                print(nD)
##                print(VT.T)
##                self.sigma_inverse[_] = np.dot(VT.T, nD).dot(U.T)
##                
##                print(np.dot(self.sigma_inverse[_],self.sigma_estimate[_]))
##                print(_)
##                print(".......")
##                print(self.sigma_inverse[_])
##        try:
##            self.sigma_inverse = {0: np.linalg.inv(self.sigma_estimate[0]),
##                                  1: np.linalg.inv(self.sigma_estimate[1]),
##                                  2: np.linalg.inv(self.sigma_estimate[2])}
##        except:
##            for i in range (self.classes):
##                print(self.sigma_estimate[i])
##            sys.exit()
##        print("________________________")
##        print(self.sigma_inverse)
##        print(np.dot(self.sigma_inverse[1],self.sigma_estimate[1]))
        # The inverse of the covariance matrix is ready

        return self.test_model_MLE()
        
        
##        print(self.det_class(self.data[2][49]))
    def test_model_MLE(self):
        # Now we will test the model on the test data
        self.testData = {0: self.data[0][self.num_of_data_points_used_for_training:self.classSize],
                         1: self.data[1][self.num_of_data_points_used_for_training:self.classSize],
                         2: self.data[2][self.num_of_data_points_used_for_training:self.classSize]}
##        print(len(self.testData[0]))
        l = self.generate_confusion_matrix()
##        self.tabulate_confusion_matrix(l)
        return self.calc_balancedAccuracy(l)

    def final_test(self):
        # Now we will test the model on the test data
        self.testData = {0: self.data[0][:self.classSize],
                         1: self.data[1][:self.classSize],
                         2: self.data[2][:self.classSize]}
##        print(len(self.testData[0]))
        l = self.generate_confusion_matrix()
        self.tabulate_confusion_matrix(l)
        return self.calc_balancedAccuracy(l)

    
    def generate_sigma_inv(self):
        self.sigma_inverse = {}
        for _ in range(self.classes):
            try:
                self.sigma_inverse[_] = np.linalg.inv(self.sigma_estimate[_])
            except:
##                a = np.linalg.eig(self.sigma_estimate[_])
##                val = a[0]
##                vec = a[1].T
##                m = np.array([[0]*len(val)]*len(val))
##                for i in range(len(val)):
##                    if val[i] != 0:
##                        
##                        m += np.multiply([1/val[i]],np.outer(vec[i], vec[i]))
##                print(m)

                
                for i in range(self.features):
                    if abs(self.sigma_estimate[_][i][i]) == 0:
                        self.sigma_estimate[_][i][i] = 10**(-6)
##                    print(self.sigma_estimate[_])
                self.sigma_inverse[_] = np.linalg.inv(self.sigma_estimate[_])
            


    # Now we will define the discriminant function  
    def det_class(self, v):
        l = []
        if self.sigma_case == 3:
            for i in range(len(self.mean_estimate)):
                Wi = np.multiply(-0.5,self.sigma_inverse[i])
                wi = np.dot(self.sigma_inverse[i], self.mean_estimate[i])
                wi0 = np.multiply(-0.5, np.dot(self.mean_estimate[i], self.sigma_inverse[i]).dot(self.mean_estimate[i]))
                wi0 += -0.5*math.log(abs(np.linalg.det(self.sigma_estimate[i])))

                g = np.dot(v, Wi).dot(v) + np.dot(wi.T,v) + wi0
                l.append(g)
        elif self.sigma_case == 1:
            for i in range(len(self.mean_estimate)):
                a = 0
                for j in range(self.features):
                    a += (v[j]-self.mean_estimate[i][j])**2
                l.append(-a)

        elif self.sigma_case == 2:
            matrix = self.sigma_estimate[0]
            for u in range(1, len(self.sigma_estimate)):
                matrix += self.sigma_estimate[u]
            matrix = np.multiply(1/len(self.sigma_estimate), matrix)
            matrix_inv = np.linalg.inv(matrix)
            for i in range(len(self.mean_estimate)):
                wi = np.dot(matrix_inv, self.mean_estimate[i])
                wi0 = np.multiply(-0.5, np.dot(self.mean_estimate[i], matrix_inv).dot(self.mean_estimate[i]))
                g = np.dot(wi.T,v) + wi0
                l.append(g)
        
        else:
            while True:
                matrix = np.random.randint(low=1, high=50, size=(self.features, self.features))  # Generate random integers between 1 and 10
                det = np.linalg.det(matrix)
                if det != 0:  # Check if the determinant is non-zero
                    break
            matrix = np.dot(matrix, matrix.T)
            matrix_inv = np.linalg.inv(matrix)
            for i in range(len(self.mean_estimate)):
                wi = np.dot(matrix_inv, self.mean_estimate[i])
                wi0 = np.multiply(-0.5, np.dot(self.mean_estimate[i], matrix_inv).dot(self.mean_estimate[i]))
                g = np.dot(wi.T,v) + wi0
                l.append(g)
                
            
##        print(l)
        return l.index(max(l))
        
    
        
    def generate_confusion_matrix(self):
        l = np.array([[0]*4]*4)
##        print(l)
        for i in range(self.classes):
            for u in self.testData[i]:
                j = self.det_class(u)
                l[i][j] += 1
            l[i][-1] = sum(l[i][0:-1])
##        print(l)
##        for i in range(self.classes):
        l[-1] = np.sum(l,axis=0) 
##        print(l)
            
        return l

    def tabulate_confusion_matrix(self,l):
        j = []
        for u in l:
            j.append(list(u))
        l = j
##        print(l)
        head = ["True Class", "w1^", "w2^", "w3^", "Total"]
        for i in range(self.classes):
            l[i].insert(0, "w"+str(i+1))
        l[-1].insert(0, "Total")
        print(tabulate(l, headers=head, tablefmt="grid"))

    def calc_balancedAccuracy(self, l):
        a = 0
        for i in range(self.classes):
##            try:
##                a += l[i][i]/(l[-1][i]*self.classes)
##                print(l[-1][i])
##            except:
##                print(l[-1][i])
            a += l[i][i]/(l[-1][i]*self.classes)
        return a
        
        



    def plot_2d(self, feature1 = 0, feature2 = 1):
        x1 = self.iris_x[:, feature1]
        print(x1)
        x2 = self.iris_x[:, feature2]

        x1_min, x1_max = x1.min() - 0.5, x1.max() + 0.5
        x2_min, x2_max = x2.min() - 0.5, x2.max() + 0.5

        plt.figure(2, figsize=(8, 6))
        plt.clf()

        plt.scatter(x1[0:50], x2[0:50], color="orange")
        plt.scatter(x1[50:100], x2[50:100], color="purple")
        plt.scatter(x1[100:150], x2[100:150], color="green")

        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)

        plt.xlabel(self.data_names[feature1])
        plt.ylabel(self.data_names[feature2])

        plt.text(x1_max - 0.4, x2_min + (x2_max - x2_min)/8, self.target_names[0], color="orange")
        plt.text(x1_max - 0.4, x2_min + (x2_max - x2_min)/8 - (x2_max - x2_min)/20, self.target_names[1], color="purple")
        plt.text(x1_max - 0.4, x2_min + (x2_max - x2_min)/8 - 2* (x2_max - x2_min)/20, self.target_names[2], color="green")
        plt.show()

    def plot_3d(self, feature1 = 0, feature2 = 1, feature3 = 2):
        x1 = self.iris_x[:, feature1]
        x2 = self.iris_x[:, feature2]
        x3 = self.iris_x[:, feature3]

        x1_min, x1_max = x1.min(), x1.max()
        x2_min, x2_max = x2.min(), x2.max()
        x3_min, x3_max = x3.min(), x3.max()


        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")
        ax.scatter3D(x1[0:50], x2[0:50], x3[0:50], color="orange")
        ax.scatter3D(x1[50:100], x2[50:100], x3[50:100], color="purple")
        ax.scatter3D(x1[100:150], x2[100:150], x3[100:150], color="green")

        ax.set_xlabel(self.data_names[feature1], fontweight='bold')
        ax.set_ylabel(self.data_names[feature2], fontweight='bold')
        ax.set_zlabel(self.data_names[feature3], fontweight='bold')

        ax.text(x1_max - 0.6, x2_min + 0.6,x3_min + 0.6, self.target_names[0], color="orange")
        ax.text(x1_max - 0.35, x2_min + 0.35, x3_min + 0.35, self.target_names[1], color="purple")
        ax.text(x1_max - 0.1, x2_min + 0.1, x3_min + 0.1, self.target_names[2], color="green")
        plt.show()

    def plot_4d(self):
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")
        temp = ax.scatter3D(self.iris_x[0:150,0], self.iris_x[0:150,1],
                           self.iris_x[0:150,2], c = self.iris_x[0:150,3])
        plt.clf()
        plt.close()

        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")

        img = ax.scatter3D(self.iris_x[0:50,0], self.iris_x[0:50,1],
                           self.iris_x[0:50,2], c = self.iris_x[0:50,3], marker = "o")

        img2 = ax.scatter3D(self.iris_x[50:100,0], self.iris_x[50:100,1],
                           self.iris_x[50:100,2], c = self.iris_x[50:100,3],
                           marker="s")

        img3 = ax.scatter3D(self.iris_x[100:150,0], self.iris_x[100:150,1],
                           self.iris_x[100:150,2], c = self.iris_x[100:150,3],
                           marker="<")

        ax.set_xlabel(self.data_names[0], fontweight='bold')
        ax.set_ylabel(self.data_names[1], fontweight='bold')
        ax.set_zlabel(self.data_names[2], fontweight='bold')
        cbar = fig.colorbar(temp)
        cbar.ax.set_ylabel(self.data_names[3], fontweight='bold')
        plt.show()


def main():
    iris = Iris()
##    iris.plot_2d()
##    iris.plot_3d()
##    iris.plot_4d()
main()
