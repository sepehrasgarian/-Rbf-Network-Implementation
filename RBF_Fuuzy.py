import pandas as pd
import csv
from collections import defaultdict
from collections import Counter
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn.model_selection import train_test_split

def distance(marakez, sa, number):
    # if(number<3):
    # print("masssssssssssssss")
    #   print(marakez)
    #  print(sa)
    b = marakez[0] - sa[0]
    c = marakez[1] - sa[1]
    e = pow(b, 2) + pow(c, 2)

    return np.sqrt(e)
def distance_array(marakez, sa):
    b = marakez[0] - sa[0]
    c = marakez[1] - sa[1]
    return [[b],[c]]
    


def power(my_list):
    # print("ghhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
    # print(x ** 2 for x in my_list)

    return [x ** 2 for x in my_list]


if __name__ == '__main__':
    numberofm=2
    numberoftheclass=4
    omega_y=0.1 
    
    clusternum = 7
    numdim, numdim2=2,2
    array = []
    fakearray = []

    df = pd.read_csv('4clstrain1200.csv')
    df = df.dropna()
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:2].values, df.iloc[:,2].values, test_size=0.3, random_state=42)
    X_train
    XandY=np.zeros((len(X_train),3))
    XandY[:,:2]=X_train
    XandY[:,2]=y_train
    
    salam=pd.DataFrame(XandY)
    sa = salam.iloc[:, 0:3].values
    for i in range(clusternum):  # 2 is number of cluster in here
        array.append([random.uniform(0, 1) for x in range(len(sa))])
    flag = 0
    for i in range(clusternum):  # 2 is number of cluster in here
        fakearray.append([0 for x in range(len(sa))])
    number = 0
    while (flag != 1):
        # for i in range(1) :
        print(number)
        number = number + 1
        for i in range(clusternum):
            for j in range(len(sa)):
                fakearray[i][j] = array[i][j]

        p = []
        ci = []
        marakez = []

        for i in range(len(array)):
            plus = []
            p = array[i]
            ci = power(p)
            plus = ci * salam.iloc[:, 0].values
            # print(plus)
            ped = sum(plus)

            v1 = ped / sum(power(p))
            v2 = sum(ci * salam.iloc[:, 1].values)
            v2 = v2 / sum(power(p))
            marakez.append(np.array([v1, v2]))
        distacearray1 = []
        for i in range(len(sa)):
            arrayofdistance = []
            for j in range(len(array)):  # number of cluster
                # print("ss")
                arrayofdistance.append(distance(marakez[j], sa[i], j))
            distacearray1.append(arrayofdistance)
        for j in range(len(sa)):
            for i in range(len(marakez)):
                sumofeleevy = 0
                for t in range(len(marakez)):
                    # print("j i t")
                    # print(j,i,t)
                    # print(distacearray1[j][i])
                    # print(distacearray1[j][t])
                    # print( array[i][j])
                    sumofeleevy += pow(distacearray1[j][i] / distacearray1[j][t],2)
                    array[i][j] = sumofeleevy
                array[i][j] = 1 / array[i][j]
        arrayofdifrence = []
        for i in range(len(marakez)):  # 2 is number of cluster in here
            arrayofdifrence.append([0 for x in range(len(sa))])
        for i in range(len(marakez)):
            for j in range(len(sa)):
                arrayofdifrence[i][j] = array[i][j] - fakearray[i][j]

        if (np.amax(arrayofdifrence) < 0.01):
            flag = 1
    plt.scatter(salam.iloc[:, 0].values,salam.iloc[:, 1].values)
    for i in range(len(marakez)):
         plt.scatter(marakez[i][0],marakez[i][1])
    plt.show()
    distance_total_array=[]
    
   
    
    for j in range(len(array)): 
        arrayofci = []
        sumarrayofci=np.zeros([numdim, numdim2])
        d=[]
        for i in range(len(sa)):# number of cluster
                # print("ss")
                arrayofci.append(np.dot(distance_array(marakez[j], sa[i]),np.transpose(distance_array(marakez[j], sa[i])))*pow(array[j][i],numberofm))
        arrayofci[j]=arrayofci[j]/sum(power(array[j]))
        for index in range(len(sa)):
            sumarrayofci+=arrayofci[index]
            
            
        distance_total_array.append(sumarrayofci) 
        
        
        
##############################################        
    
    distance_totalllll_array=[]
    distance_totalllll_array_transpose=[]
    for j in range(len(array)): 
        arrayofRBF = []
        d=[]
        for i in range(len(sa)):# number of cluster
                # print("ss")
                arrayofRBF.append(np.dot(np.dot(np.transpose(distance_array(marakez[j], sa[i])) ,inv(distance_total_array[j]) ),distance_array(marakez[j], sa[i])))
                    
        my_new_list = [-p * omega_y for p in arrayofRBF]
        distance_totalllll_array_transpose.append( [np.exp(t) for t in my_new_list])       
       
    distance_totalllll_array=np.transpose(distance_totalllll_array_transpose).reshape(len(sa),len(array))
    Weight=[]
    distance_totall_array_transpse_new=[]
    distance_totall_array_transpse_new=np.transpose(distance_totalllll_array)
    gtg=np.dot(distance_totall_array_transpse_new,distance_totalllll_array)
    inverseg=inv(gtg)
    matrixweigh=np.dot(inverseg,distance_totall_array_transpse_new)
    y_matrix=[]
    for i in range(len(sa)):  # 2 is number of cluster in here
        y_matrix.append([0 for x in range(numberoftheclass)])
        
    for i in range (len(sa)):
       if (sa[i][2]==-1):
           sa[i][2]=2
          
       p=sa[i][2]
       
       y_matrix[i][int(p-1)]=1
       
    weight=[]
    weight=np.dot(matrixweigh,y_matrix)  
    finalmatrix=np.dot(distance_totalllll_array,weight) 
    hj=[]
    hj=np.argmax(finalmatrix,axis=1)+1
    count=0
    for i in range (len(sa)):
        count+= abs(np.sign(sa[i][2]-hj[i]))
    accuracy=1-count/len(sa)
    
    plt.scatter(sa[:, 0], sa[:, 1], c=hj, s=50, cmap='viridis')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,15))
    plt.show()
    
    
#######################################
#for test    

    XandY_test = np.zeros((len(X_test), 3))
    XandY_test[:, :2] = X_test
    XandY_test[:, 2] = y_test

    XandY_test_Data_frame = pd.DataFrame(XandY_test)
    sa = XandY_test_Data_frame.iloc[:, 0:3].values
    array = []
    fakearray = []
    distacearray1 = []
    array_for_test = []
    for i in range(clusternum):  # 2 is number of cluster in here
        array_for_test.append([random.uniform(0, 1) for x in range(len(sa))])

    for i in range(len(sa)):
        arrayofdistance = []
        for j in range(len(array_for_test)):  # number of cluster
            # print("ss")
            arrayofdistance.append(distance(marakez[j], sa[i], j))
        distacearray1.append(arrayofdistance)
    for j in range(len(sa)):
        for i in range(len(marakez)):
            sumofeleevy = 0
            for t in range(len(marakez)):
                # print("j i t")
                # print(j,i,t)
                # print(distacearray1[j][i])
                # print(distacearray1[j][t])
                # print( array[i][j])
                sumofeleevy += pow(distacearray1[j][i] / distacearray1[j][t], 2)
                array_for_test[i][j] = sumofeleevy
            array_for_test[i][j] = 1 / array_for_test[i][j]
    plt.scatter(XandY_test_Data_frame.iloc[:, 0].values, XandY_test_Data_frame.iloc[:, 1].values)
    for i in range(len(marakez)):
        plt.scatter(marakez[i][0], marakez[i][1])
    plt.show()
    distance_total_array = []

    for j in range(len(array_for_test)):
        arrayofci = []
        sumarrayofci = np.zeros([numdim, numdim2])
        d = []
        for i in range(len(sa)):  # number of cluster
            # print("ss")
            arrayofci.append( np.dot(distance_array(marakez[j], sa[i]), np.transpose(distance_array(marakez[j], sa[i]))) * pow(array_for_test[j][i], numberofm))
        arrayofci[j] = arrayofci[j] / sum(power(array_for_test[j]))
        for index in range(len(sa)):
            sumarrayofci += arrayofci[index]

        distance_total_array.append(sumarrayofci)

    distance_totalllll_array = []
    final_array_of_matrix = []
    new_transpose = []
    for j in range(len(array_for_test)):
        arrayofRBF = []
        d = []
        for i in range(len(sa)):  # number of cluster
            # print("ss")
            arrayofRBF.append(
                np.dot(np.dot(np.transpose(distance_array(marakez[j], sa[i])), inv(distance_total_array[j])),
                       distance_array(marakez[j], sa[i])))
        my_new_list = [-p * omega_y for p in arrayofRBF]
        new_transpose.append([np.exp(t) for t in my_new_list])
    distance_totalllll_array=np.transpose(new_transpose).reshape(len(sa),len(array_for_test))    
    final_array_of_matrix = np.dot(distance_totalllll_array, weight)

    count = 0
    accuracy_test = 0
    hja=np.argmax(final_array_of_matrix,axis=1)+1
    
    
    for i in range(len(sa)):
         count += abs(np.sign(sa[i][2] - hja[i]))
         accuracy_test = 1 - count / len(sa)

    plt.scatter(sa[:, 0], sa[:, 1], c=hja, s=50, cmap='viridis')

    plt.figure(figsize=(15, 15))
    plt.show()
        
        
       

    
    
    
   
    
    
    
#################################################    
            
        
    
          

    











