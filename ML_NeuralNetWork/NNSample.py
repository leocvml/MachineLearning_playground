
# LoadData -> (Norm_Data) -> train(error rate , ) -> test
# sigmoid(inputData * weight )

from numpy import *

def Load_Data():
    Training_Input = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    Traing_Output = array([[0,1,1,0]]).T
    return Training_Input , Traing_Output

def Norm_Data(data):
    minVal = data.min(0)
    maxVal = data.max(0)
    ranges = maxVal - minVal
    normData = zeros(shape(data))
    m = data.shape[0]
    normData = data - tile(minVal , (m,1))
    normData = normData / tile(ranges , (m,1))
    return normData

def weight():
    random.seed(1)
    nerual_weight = 2 * random.random((3,1)) - 1
    return nerual_weight

def act_sigmoid(x):
    return 1 / (1 + exp(-x))

def sig_der(x):
    return x * (1 - x)


def train(input_data , output_data , it):
    nerual_weight = weight()
    for i in range(it):                
        out = act_sigmoid(dot(input_data,nerual_weight))
        error = output_data - out
        temp = error * sig_der(out)
        adjust = dot(input_data.T , temp)
        nerual_weight += adjust
    return nerual_weight

def test(input_data,weight):
    return act_sigmoid(dot(input_data,weight))
    
