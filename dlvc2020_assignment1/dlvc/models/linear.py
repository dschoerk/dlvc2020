
from ..model import Model

import numpy as np
import torch
import torch.nn as nn

class LinearClassifier(Model):
    '''
    Linear classifier without bias.
    Returns softmax class scores (see lecture slides).
    '''

    def __init__(self, input_dim: int, num_classes: int, lr: float, momentum: float, nesterov: bool):
        '''
        Ctor.
        input_dim is the length of input vectors (> 0).
        num_classes is the number of classes (> 1).
        lr: learning rate to use for training (> 0).
        momentum: momentum to use for training (> 0).
        nesterov: training with or without Nesterov momentum.
        '''

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.momentum = momentum
        self.nesterov = nesterov
        self.lr = lr
        
        self.weights = torch.randn(input_dim, num_classes, requires_grad=True, dtype=torch.float)
        self.v = torch.zeros(self.weights.size())

        # TODO implement

    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple, which is (0, input_dim).
        '''

        return (0, self.input_dim)

    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''

        return (self.num_classes, 0)

        # TODO implement

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on batch of data.
        Data are the input data, with shape (m, input_dim) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns the current cross-entropy loss on the batch.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        #if data.shape[1] != self.input_shape()[1]:
        #    raise TypeError()

        if data.dtype != np.float32:
            raise TypeError()

        if labels.shape[0] != data.shape[0]:
            raise TypeError()

        if (labels < 0).any() or (labels >= self.num_classes).any():
            raise ValueError()

        try:

            #print(data.shape)
            #print(labels.shape)

            #print(data.shape)
            #print(labels.shape)
            
            loss = nn.CrossEntropyLoss()
            input = torch.tensor(data, dtype=torch.float)
            scores = torch.mm(input, self.weights)
            #pred = scores.argmax(dim=0)
            #print(pred)
            
            #print(scores.shape)
            

            labels = torch.tensor(labels, dtype=torch.long)
            #print(labels)
            #print(labels.shape)
            
            # apply softmax to pred here?
            output = loss(
                scores, 
                labels)
            output.backward()

            grad = self.weights.grad
            #print("gradient: %s" % str(grad))
            print("loss %f" % output.data)

            # TODO implement (compute loss)

            # self.weights.retain_grad() # include this tensor in the computation graph
            # loss.backward() # compute gradients with backpropagation

            # TODO implement (update weights with gradient descent)

            # compute velocity
            grad = grad / grad.norm()
            #print(grad)
            self.v = self.v * self.momentum - grad * self.lr
            #self.v = -grad
            
            # update weights
            self.weights.data = self.weights + self.v
            
            return output.data
        
        except Exception as e:
            raise RuntimeError(str(e))


        

    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict softmax class scores from input data.
        Data are the input data, with a shape compatible with input_shape().
        The label array has shape (n, output_shape()) with n being the number of input samples.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        input = torch.tensor(data, dtype=torch.float)
        scores = torch.mm(input, self.weights)
        #print(scores)
        sm = nn.Softmax(dim=1)
        scores = sm(scores)
        #print(scores)
        return scores.data
        #exit(0)
        



        # TODO implement