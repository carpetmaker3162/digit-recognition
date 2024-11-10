'''
usage: python3 load_weights.py | pbcopy
(run from same directory as model.weights.h5 file)

then paste into a python file
'''

import numpy as np
from tensorflow.keras.models import load_model
import sys

def recursive_round(obj, precision):
    if precision is None:
        return obj

    if isinstance(obj, list):
        return [recursive_round(item, precision) for item in obj]
    elif isinstance(obj, float):
        return round(obj, precision)
    else:
        return obj

def prune(s):
    return s.replace(' ', '').replace('0.', '.')

if __name__ == '__main__':
    model = load_model('model.weights.h5')
    num_layers = len(model.layers) - 1
    print(num_layers, 'layers', file=sys.stderr)
    
    for i in range(1, num_layers + 1):
        weights = model.layers[i].weights[0]
        biases = model.layers[i].bias
    
        print('weights_{} = {}\n'.format(i, 
            prune(
                str(
                    recursive_round(np.array(weights).tolist(), 4)
                )
            )
        ))
        print('biases_{} = {}\n'.format(i, 
            prune(
                str(
                    recursive_round(np.array(biases).tolist(), 4)
                )
            )
        ))