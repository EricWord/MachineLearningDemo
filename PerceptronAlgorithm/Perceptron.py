import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt
class Perceptron (object):


    def  __init__(self,eta=0.01,n_iter=10):
        self.eta=eta;
        self.n_iter=n_iter

    def fit(self, X,  y):
        self.w_=np.zeros(1+X.shape[1])
        self.errors_=[]

        for _ in range(self.n_iter):
            errors=0
            for xi ,target in zip(X,y):
                update=self.eta*(target - self.predict(xi))
                self.w_[1:]+=update*xi
                self.w_[0]+=update
                errors+=int(update!=0.0)
                self.errors_.append(errors)
        return self


    def net_input(self,X):
        return  np.dot(X,self.w_[1:])+self.w_[0]

    def predict(self,X):
        return  np.where(self.net_input(X)>=0.0,1,-1)




if __name__=="__main__":

    df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
    print(df.tail())
    y=df.iloc[0:100,4].values
    y=np.where(y=='Iris-setosa',-1,1)
    x=df.iloc[0:100,[0,2]].values
    plt.scatter(x[:50,0],x[:50,1],color='red',marker='o',label='setosa')
    plt.scatter(x[50:100,0],x[50:100,1],color='blue',marker='x',label='versicolocr')
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')
    plt.show()