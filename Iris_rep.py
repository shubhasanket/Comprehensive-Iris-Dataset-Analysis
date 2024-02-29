from sklearn import datasets
import matplotlib.pyplot as plt

DEBUG = True

class Iris:
    def __init__(self):
        iris = datasets.load_iris()
        print(iris)
        self.data_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
        self.target_names = iris['target_names']
        if DEBUG: print(self.target_names)
        self.iris_x = iris['data']
        if DEBUG: print(self.iris_x)
        self.iris_y = iris['target']
        if DEBUG: print(self.iris_y)        

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

    def plot_3d(self, feature1 = 1, feature2 = 2, feature3 = 3):
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
##        temp = ax.scatter3D(self.iris_x[0:150,0], self.iris_x[0:150,1],
##                           self.iris_x[0:150,2], c = self.iris_x[0:150,3])
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
    iris.plot_3d(0,1,2)
    iris.plot_3d(1,3,2)
    iris.plot_3d(0,3,2)
    iris.plot_4d()
main()
