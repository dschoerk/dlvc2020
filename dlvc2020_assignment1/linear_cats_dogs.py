from collections import namedtuple

from dlvc.models.linear import LinearClassifier
from dlvc.test import Accuracy

from dlvc.datasets.pets import PetsDataset, Subset
from dlvc.batches import BatchGenerator

import numpy as np

TrainedModel = namedtuple('TrainedModel', ['model', 'accuracy'])

path = "..\cifar-10-python\cifar-10-batches-py"

# 1) Load the training, validation, and test sets as individual PetsDatasets.
train = PetsDataset(path, Subset.TRAINING)
test = PetsDataset(path, Subset.TEST)
val = PetsDataset(path, Subset.VALIDATION)

# 2) Create a BatchGenerator for each one.
batchsize = 100
train_batches = BatchGenerator(train, batchsize, shuffle=True)
test_batches = BatchGenerator(test, batchsize, shuffle=False)
val_batches = BatchGenerator(val, batchsize, shuffle=False)



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

            data = batch.data
            data = np.reshape(data, (data.shape[0], 32*32*3)).astype(np.float32) ## REMOVE must be given by batch gen

            """import cv2
            im = np.reshape(data[0, :], (32, 32, 3))
            print(im)
            cv2.imshow("", cv2.resize(im / 255, (128, 128)))
            cv2.waitKey()"""

            clf.train(data, batch.label)

            #print(batch.data.dtype)
            #print(data.shape)
            #print(batch.label.shape)

    accuracy = Accuracy()
    for batch in val_batches:
        data = batch.data
        data = np.reshape(data, (data.shape[0], 32*32*3)).astype(np.float) ## REMOVE must be given by batch gen

        pred = clf.predict(data)
        accuracy.update(pred, batch.label)

        # predict and update accuracy

    return TrainedModel(clf, accuracy)




model = train_model(lr = 0.01, momentum = 0.5)
print(model.accuracy)

# TODO implement steps 4-7