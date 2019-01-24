from numpy import *

def file2array_norm(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    data_in =[]
    data_out =[]
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        data_in.append([float(listFromLine[0]),float(listFromLine[1]),float(listFromLine[2])])
        data_out.append([int(listFromLine[3])])
    fr.close()
    
    data_in = array(data_in)
    data_out = (array(data_out))  
    minVals = data_in.min(0)
    maxVals = data_in.max(0)
    print(minVals)
    print(maxVals)
    range = maxVals - minVals;
    normDataSet = zeros(shape(data_in))
    m = data_in.shape[0]
    normDataSet = data_in - tile(minVals , (m,1))
    normDataSet = normDataSet / tile(range,(m,1))
    print(normDataSet[0:8])


    fr = open('normDataSet.txt')
    lenth = normDataSet.shape[0]
    print(lenth)
    for i in range(lenth):
        print(str(normDataSet[i]))
        

    fr.close()

    return normDataSet , data_out

