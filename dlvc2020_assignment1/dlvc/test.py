import numpy as np

from abc import ABCMeta, abstractmethod


class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass

    @abstractmethod
    def __lt__(self, other) -> bool:
        '''
        Return true if this performance measure is worse than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass

    @abstractmethod
    def __gt__(self, other) -> bool:
        '''
        Return true if this performance measure is better than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass


class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self):
        '''
        Ctor.
        '''

        # self.update_count = 0.
        # self.accuracy_value = 0.
        self.reset()

    def reset(self):
        '''
        Resets the internal state.
        '''
        self.update_count = 0.
        self.accuracy_value = 0.

    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''

        if prediction.shape[0] != target.shape[0]:
            raise ValueError("Invalid dimensions!")

        if np.min(target) < 0 or np.max(target) >= prediction.shape[1]:
            raise ValueError("Invalid labels in ground truth!")

        num_samples = prediction.shape[0]

        # compare labels with diff, abs --> false pred > 0, true pred = 0, min to map x > 0 -> x=1, 1-x to inverse it --> number of correctly classified labels)
        correct_samples = sum([1. - np.minimum(1, np.abs(np.argmax(prediction[idx]) - target[idx])) for idx in range(num_samples)])
        
        # vorschlag für bessere lesbarkeit
        # correct_samples = np.sum(np.argmax(prediction, axis=1) == target[idx]) 
        
        acc = correct_samples / num_samples

        # incremental average
        self.update_count += 1
        self.accuracy_value = self.accuracy_value + ((acc - self.accuracy_value) / (1. * self.update_count))

    def __str__(self):
        '''
        Return a string representation of the performance.
        '''

        return "accuarcy: %f" % self.accuracy_value
        # return something like "accuracy: 0.395"

    def __lt__(self, other) -> bool:
        '''
        Return true if this accuracy is worse than another one.
        Raises TypeError if the types of both measures differ.
        '''

        return self.accuracy_value < other

    def __gt__(self, other) -> bool:
        '''
        Return true if this accuracy is better than another one.
        Raises TypeError if the types of both measures differ.
        '''

        return self.accuracy_value > other

    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''

        return self.accuracy_value
