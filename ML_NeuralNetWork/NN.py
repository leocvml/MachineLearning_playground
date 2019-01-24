from numpy import exp, array, random, dot
from numpy import *

 

class NeuralNetWork():
    def __init__(self):

        random.seed(1)
        self.synaptic_weights = 2 * random.random((3,1)) - 1
        
    def __sigmoid(self,x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self,x):
        return x * (1-x)

    def train(self , train_set_input ,train_set_output , number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            #print(iteration)
            
            output =self.think(train_set_input)
            error = train_set_output - output
            temp = error * self.__sigmoid_derivative(output)
            adjustment = dot(train_set_input.T , temp )
            self.synaptic_weights += adjustment
            
    def think(self , inputs): #(self , inputs)
        return self.__sigmoid(dot(inputs,self.synaptic_weights)) #self.synaptic_weights

    
def auto_norm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals , (m,1))
    normDataSet = normDataSet / tile(ranges , (m,1))
    return normDataSet


def file2array(filename):
    fr = open(filename)   #filename
    numberOfLines = len(fr.readlines())
    data_in =[]
    data_out =[]
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        data_in.append([float(listFromLine[0]),float(listFromLine[1]),float(listFromLine[2])])
        data_out.append([int(listFromLine[3])])
    data_in = array(data_in)
    data_out = (array(data_out))
    return data_in , data_out

def train(data_in,data_out,it):
    neural_network = NeuralNetWork()
    print ("Random Starting synaptic weights : ")
    print (neural_network.synaptic_weights)
    neural_network.train(data_in,data_out,it)
    print ("new synaptic weights after training")
    print (neural_network.synaptic_weights)
    return neural_network.synaptic_weights

def sig(x):
    return 1 / (1 + exp(-x))
    
def test(weight,a,b,c):
    neural_network = NeuralNetWork()
    inputs = array([a,b,c])
    print(sig(dot(inputs,weight)))
    
  
    
def sample():    
        neural_network = NeuralNetWork()
        print ("Random Starting synaptic weights : ")
        print (neural_network.synaptic_weights)
        train_set_input = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])  #[[0,0,1],[1,1,1],[1,0,1],[0,1,1]]
        train_set_output = array([[0,1,1,0]]).T         #[[0,1,1,0]]
        neural_network.train(train_set_input,train_set_output,10000)
        print ("new synaptic weights after training")
        print (neural_network.synaptic_weights)
        print ("Considering new situation[1,0] -> : ")
        print (neural_network.think(array([0,0,1])))


    
   

