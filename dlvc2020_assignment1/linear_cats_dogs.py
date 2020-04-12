from collections import namedtuple

from dlvc.models.linear import LinearClassifier
from dlvc.test import Accuracy

from dlvc.datasets.pets import PetsDataset, Subset
from dlvc.batches import BatchGenerator

import dlvc.ops as ops

import numpy as np

TrainedModel = namedtuple('TrainedModel', ['model', 'accuracy'])

path = "..\cifar-10-python\cifar-10-batches-py"

# 1) Load the training, validation, and test sets as individual PetsDatasets.
train = PetsDataset(path, Subset.TRAINING)
test = PetsDataset(path, Subset.TEST)
val = PetsDataset(path, Subset.VALIDATION)

# 2) Create a BatchGenerator for each one.


op = ops.chain([
    ops.vectorize(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1/127.5),
])

batchsize = 256
train_batches = BatchGenerator(train, batchsize, shuffle=False, op=op)
test_batches = BatchGenerator(test, batchsize, shuffle=False, op=op)
val_batches = BatchGenerator(val, batchsize, shuffle=False, op=op)



def train_model(lr: float, momentum: float) -> TrainedModel:
    '''
    Trains a linear classifier with a given learning rate (lr) and momentum.
    Computes the accuracy on the validation set.
    Returns both the trained classifier and accuracy.
    '''

    # 3) Complete the function train_model. It trains a linear classifier for 10 epochs 
    # and then computes the accuracy on the validation set. You can choose to train with or without Nesterov momentum.


    # TODO implement step 3

    clf = LinearClassifier(
        input_dim = 32 * 32 * 3, 
        num_classes = 2, 
        lr = lr, 
        momentum = momentum, 
        nesterov = False)

    n_epochs = 10
    for i in range(n_epochs):
        for batch in train_batches:
            # train classifier

            clf.train(batch.data, batch.label)

            #print(batch.data.dtype)
            #print(data.shape)
            #print(batch.label.shape)

    accuracy = Accuracy()
    for batch in val_batches:
        pred = clf.predict(batch.data)
        accuracy.update(pred, batch.label)

        # predict and update accuracy

    return TrainedModel(clf, accuracy)




model = train_model(lr = 0.1, momentum = 0.9)
print(model.accuracy)

# TODO implement steps 4-7