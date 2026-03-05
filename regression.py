import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("prix_maisons (1).csv")

print(df.head())
print(df.dtypes)
print("Nombre de lignes :", len(df))


x = df["surface"].values
y = df["prix"].values


x = (x - x.mean()) / x.std()
y = (y - y.mean()) / y.std()

plt.scatter(x, y)
plt.xlabel("surface (std)")
plt.ylabel("prix (std)")
plt.title("Surface vs Prix")
plt.show()



def quadratic(a,b,c,x):
    return a*x**2 + b*x + c



def mse(y_hat,y):
    return np.mean((y_hat-y)**2)



def rmse(y_hat,y):
    return np.sqrt(mse(y_hat,y))



def train(x,y,lr=0.05,epochs=300):

    a = np.random.randn()*0.1
    b = np.random.randn()*0.1
    c = np.random.randn()*0.1

    n = len(x)

    errors=[]

    for epoch in range(epochs):

        y_hat = quadratic(a,b,c,x)
        e = y_hat-y

        da = (2/n)*np.sum(e*x**2)
        db = (2/n)*np.sum(e*x)
        dc = (2/n)*np.sum(e)

        a = a - lr*da
        b = b - lr*db
        c = c - lr*dc

        errors.append(np.sqrt(np.mean(e**2)))

        if epoch%50==0:
            print("epoch",epoch,"RMSE",errors[-1])

    return a,b,c,errors


a,b,c,errors = train(x,y)


x_line = np.linspace(min(x),max(x),200)
y_line = quadratic(a,b,c,x_line)

plt.scatter(x,y)
plt.plot(x_line,y_line)
plt.title("Regression quadratique")
plt.show()



plt.plot(errors)
plt.xlabel("epochs")
plt.ylabel("RMSE")
plt.title("Evolution de la RMSE")
plt.show()