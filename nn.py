from weights import *

def flatten(array_2d):
    '''flatten a 2d array'''
    result = []
    for row in array_2d:
        for cell in row:
            result.append(cell)
    return result

def predict(flattened):
    '''input flattened 28x28 bitmap image'''
    SZ_1 = len(biases_1)
    SZ_2 = len(biases_2)
    SZ_3 = len(biases_3)
    
    layer1 = [0 for _ in range(64)]
    layer2 = [0 for _ in range(48)]
    layer3 = [0 for _ in range(10)]

    for i in range(SZ_1):
        for j in range(28*28):
            layer1[i] += flattened[j] * weights_1[j][i]
        layer1[i] += biases_1[i]
        layer1[i] = max(0, layer1[i])

    for i in range(SZ_2):
        for j in range(SZ_1):
            layer2[i] += layer1[j] * weights_2[j][i]
        layer2[i] += biases_2[i]
    
    for i in range(SZ_3):
        for j in range(SZ_2):
            layer3[i] += layer2[j] * weights_3[j][i]
        layer3[i] += biases_3[i]

    return layer3