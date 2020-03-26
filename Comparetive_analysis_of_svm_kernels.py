# importing the required packages 
import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates =[]
prices=[]
# Function for reading file 
def get_data(filename):
    with open(filename,'r')as csvfile:
        csvFileReader=csv.reader(csvfile)
        next (csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('/')[1]))
            prices.append(float(row[1]))
    return

get_data('Training.csv')
x=[[30]]# date for which the open price is predicted
dates=np.reshape(dates,(len(dates),1))
#instantiating kernels
svr_lin=SVR(kernel='linear',C=1e3)
svr_poly=SVR(kernel='poly',C=1e3,degree=2)
svr_rbf=SVR(kernel='rbf',C=1e3,gamma='scale')
#fitting the data in to kernels
svr_lin.fit(dates,prices)
svr_poly.fit(dates,prices)
svr_rbf.fit(dates,prices)
#predicting the open price for the date '30' using predict fn
predicted_price=[svr_lin.predict(x)[0],svr_poly.predict(x)[0],svr_rbf.predict(x)[0]]
print ("linear model :",svr_lin.predict(x)[0],"polynomial model :",svr_poly.predict(x)[0],"RBF :",svr_rbf.predict(x)[0])
tprices=[]
#reading test data
with open("test.csv",'r')as csvfile:
        tdata=csv.reader(csvfile)
        next (tdata)
        for row in tdata:
            tprices.append(float(row[1]))

#printing the difference in predicted and actual price
print("The actual value :",tprices)
print("Differences : ")
ldiff=abs(tprices-predicted_price[0])
pdiff=abs(tprices-predicted_price[1])
rdiff=abs(tprices-predicted_price[2])
print(ldiff,pdiff,rdiff)
#printing the error percentage
print("Error percentage:")
pldiff=int(((abs(tprices-predicted_price[0])*100))/tprices)
ppdiff=int(((abs(tprices-predicted_price[1])*100))/tprices)
prdiff=int(((abs(tprices-predicted_price[2])*100))/tprices)
print("The difference percentage of linear model :",pldiff,"%")
print("The difference percentage of polynimial model :",ppdiff,"%")
print("The difference percentage of RBF model :",prdiff,"%")

#plotting the outputs in graphs 
plt.scatter(dates,prices,color='black',label='Data')          
plt.plot(dates,svr_lin.predict(dates),color='green',label='linear model ')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('SVR- Linear Model')
plt.legend()
plt.figure()
plt.scatter(dates,prices,color='black',label='Data')
plt.plot(dates,svr_poly.predict(dates),color='blue',label='polynomial model ')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('SVR- Polynomial Model')
plt.legend()
plt.figure()
plt.scatter(dates,prices,color='black',label='Data')
plt.plot(dates,svr_rbf.predict(dates),color='red',label='RBF model ')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('SVR- RBF model')
plt.legend()
plt.figure()
#plotting all graphs in single output 
plt.scatter(dates,prices,color='black',label='Data')          
plt.plot(dates,svr_lin.predict(dates),color='green',label='linear model ')            
plt.plot(dates,svr_poly.predict(dates),color='blue',label='polynomial model ')
plt.plot(dates,svr_rbf.predict(dates),color='red',label='RBF model ')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('SVR')
plt.legend()
plt.show()
