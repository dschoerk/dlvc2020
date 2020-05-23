
from dlvc.dataset import Sample, Subset, ClassificationDataset
import os
import numpy as np
import cv2

class PetsDataset(ClassificationDataset):
    '''
    Dataset of cat and dog images from CIFAR-10 (class 0: cat, class 1: dog).
    '''

    def __init__(self, fdir: str, subset: Subset):
        '''
        Loads a subset of the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all cat and dog images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all cat and dog images from "data_batch_5".
          - The test set contains all cat and dog images from "test_batch".

        Images are loaded in the order the appear in the data files
        and returned as uint8 numpy arrays with shape 32*32*3, in BGR channel order.
        '''

        # TODO implement
        # See the CIFAR-10 website on how to load the data files

        # Aus der Angabe:
        # Number of samples in the individual datasets: 7959 (training), 2041 (validation), 2000 (test).
        # Total number of cat and dog samples: 6000 per class
        # Image shape: always (32, 32, 3, image type: always np.uint8
        # Labels of first 10 training samples: 0 0 0 0 1 0 0 0 0 1
        # Make sure that the color channels are in BGR order (not RGB) by displaying the images and verifying the colors are correct (cv2.imshow, cv2.imwrite).

        # assign cifar batch files to train/val/test sets
        files = {}
        files[Subset.TRAINING] = ["data_batch_%d" % x for x in range(1,5)]
        files[Subset.VALIDATION] = ["data_batch_5"]
        files[Subset.TEST] = ["test_batch"]

        # selection of classes
        self.selected_classes = [3, 5] # cats=3, dogs=5
        
        if not os.path.isdir(fdir):
            raise ValueError("Folder '%s' does not exist!" % fdir)

        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        # stores results from single files, concatenated later
        image_arr = []
        label_arr = []

        for f in files[subset]:
            fpath = os.path.join(fdir, f) # path to cifar file

            try:
                data = unpickle(fpath)
            except FileNotFoundError as e:
                raise ValueError("file not found")

            images = np.array([cv2.cvtColor(image.reshape((3, 32, 32)).transpose(1, 2, 0), cv2.COLOR_RGB2BGR) for image in np.array(data[b'data'])])

            # decode labels and create cat/dog mask
            labels = np.array(data[b'labels'])
            
            mask = np.zeros_like(labels)
            labels_new = np.zeros_like(labels)
            for i, c in enumerate(self.selected_classes):
                mask = np.logical_or(labels == c, mask)
                labels_new[labels == c] = i

            # apply mask for selected classes            
            labels_new = labels_new[mask]
            images = images[mask, :, :, :]

            image_arr.append(images)
            label_arr.append(labels_new)

        # concatenate results from single files
        self.images = np.concatenate(image_arr, axis=0)
        self.labels = np.concatenate(label_arr, axis=0)

        #### show images ####
        if False:
            print(labels[0:10])
            for idx in range(images.shape[0]):
                vis = cv2.resize(images[idx, :, :, :], (128, 128))
                cv2.imshow("", vis) # imshow uses BGR so images should be fine
                cv2.waitKey()
        ######################


    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''

        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds.
        ''' 

        image = self.images[idx, :, :, :]
        label = self.labels[idx]

        # deactivated asserts to allow slicing
        #assert(image.shape == (32,32,3))
        #assert(image.dtype == np.uint8)
        #assert(label == 0 or label == 1)

        return Sample(
            idx=idx, 
            data=image, 
            label=label)

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''

        return len(self.selected_classes)


##### Validation Checks #####
if True:
    # path = "E:\TU\dlvc\cifar-10-python\cifar-10-batches-py"
    path = "..\\cifar-10-python\\cifar-10-batches-py"
    train = PetsDataset(path, Subset.TRAINING)
    test = PetsDataset(path, Subset.TEST)
    val = PetsDataset(path, Subset.VALIDATION)

    assert(len(train) == 7959)
    assert(len(val) == 2041)
    assert(len(test) == 2000)
    assert((train[0:10].label == [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).all())

    # 6000 samples per class
    assert((train[:].label == 0).sum()+(val[:].label == 0).sum()+(test[:].label == 0).sum() == 6000)
    assert((train[:].label == 1).sum()+(val[:].label == 1).sum()+(test[:].label == 1).sum() == 6000)
#############################