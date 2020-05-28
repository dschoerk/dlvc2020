
from ..model import Model

import numpy as np
import torch
import torch.nn as nn

class CnnClassifier(Model):
    '''
    Wrapper around a PyTorch CNN for classification.
    The network must expect inputs of shape NCHW with N being a variable batch size,
    C being the number of (image) channels, H being the (image) height, and W being the (image) width.
    The network must end with a linear layer with num_classes units (no softmax).
    The cross-entropy loss (torch.nn.CrossEntropyLoss) and SGD (torch.optim.SGD) are used for training.
    '''

    def __init__(self, net: nn.Module, input_shape: tuple, num_classes: int, lr: float, wd: float, momentum: float):
        '''
        Ctor.
        net is the cnn to wrap. see above comments for requirements.
        input_shape is the expected input shape, i.e. (0,C,H,W).
        num_classes is the number of classes (> 0).
        lr: learning rate to use for training (SGD with e.g. Nesterov momentum of 0.9).
        wd: weight decay to use for training.
        '''

        # TODO implement

        # Inside the train() and predict() functions you will need to know whether the network itself
        # runs on the CPU or on a GPU, and in the latter case transfer input/output tensors via cuda() and cpu().
        # To termine this, check the type of (one of the) parameters, which can be obtained via parameters() (there is an is_cuda flag).
        # You will want to initialize the optimizer and loss function here.
        # Note that PyTorch's cross-entropy loss includes normalization so no softmax is required

        self.on_gpu = next(net.parameters()).is_cuda # determine if running on gpu or cpu

        self.optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.net = net

        if self.on_gpu:
            self.net = net.cuda()

    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple.
        '''

        return self.input_shape

    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''

        return (self.num_classes, )

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on batch of data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns the training loss.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        if data.dtype != np.float32:
            raise TypeError("invalid data datatype")

        if labels.dtype != np.int and labels.dtype != np.uint:
            raise TypeError("invalid label datatype")

        if data.shape[1:] != self.input_shape[1:]:
            raise ValueError("invalid data dimensions")

        #if labels.shape[0] != data.shape[0] or data.shape[1] != self.input_shape()[1]:
        #    raise TypeError("invalid input dimensions")

        if (labels < 0).any() or (labels >= self.num_classes).any():
            raise ValueError()

        # TODO implement
        # Make sure to set the network to train() mode
        # See above comments on CPU/GPU

        try:
            self.net.train()

            self.optimizer.zero_grad()

            

            input = torch.from_numpy(data).cuda()
            outputs = self.net(input).cpu()
            l = torch.from_numpy(labels).type(torch.long)

            loss = self.loss_fn(outputs, l)
            loss.backward()
            self.optimizer.step()

            return loss.detach().numpy()
        except Exception as e:
            raise RuntimeError(str(e))


    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict softmax class scores from input data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        The scores are an array with shape (n, output_shape()).
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        if data.dtype != np.float32:
            raise TypeError("invalid data datatype")

        if data.shape[1:] != self.input_shape[1:]:
            raise ValueError("invalid data dimensions")

        # TODO implement

        # Pass the network's predictions through a nn.Softmax layer to obtain softmax class scores
        # Make sure to set the network to eval() mode
        # See above comments on CPU/GPU
        try:
            self.net.eval()
            input = torch.from_numpy(data).cuda()
            outputs = self.net(input).cpu()
            sm = torch.nn.Softmax()
            pred = sm(outputs)

            return pred.detach().numpy()
        except Exception as e:
            raise RuntimeError(str(e))

        
