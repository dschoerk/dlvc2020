import cv2
import numpy as np

from typing import List, Callable

# All operations are functions that take and return numpy arrays
# See https://docs.python.org/3/library/typing.html#typing.Callable for what this line means
Op = Callable[[np.ndarray], np.ndarray]


def chain(ops: List[Op]) -> Op:
    '''
    Chain a list of operations together.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        for op_ in ops:
            sample = op_(sample)
        return sample

    return op


def type_cast(dtype: np.dtype) -> Op:
    '''
    Cast numpy arrays to the given type.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample.astype(dtype=dtype)

    return op


def vectorize() -> Op:
    '''
    Vectorize numpy arrays via "numpy.ravel()".
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return np.ravel(sample)

    return op


def add(val: float) -> Op:
    '''
    Add a scalar value to all array elements.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return np.add(sample, val)

    return op

def add(val: (float, float, float)) -> Op:
    '''
    Add a scalar value to all array elements.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return np.add(sample, val)

    return op


def mul(val: float) -> Op:
    '''
    Multiply all array elements by the given scalar.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return np.multiply(sample, val)

    return op


def mul(val: (float, float, float)) -> Op:
    '''
    Multiply all array elements by the given scalar.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return np.multiply(sample, val)

    return op


def hwc2chw() -> Op:
    '''
    Flip a 3D array with shape HWC to shape CHW.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample.transpose(2, 0, 1)

    return op


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def chw2hwc() -> Op:
    '''
    Flip a 3D array with shape CHW to HWC.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample.transpose(1, 2, 0)

    return op


def hflip() -> Op:
    '''
    Flip arrays with shape HWC horizontally with a probability of 0.5.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        if np.random.uniform() <= 0.5:
            return np.flip(sample, axis=1)
        return sample

    return op


def rcrop(sz: int, pad: int, pad_mode: str) -> Op:
    '''
    Extract a square random crop of size sz from arrays with shape HWC.
    If pad is > 0, the array is first padded by pad pixels along the top, left, bottom, and right.
    How padding is done is governed by pad_mode, which should work exactly as the 'mode' argument of numpy.pad.
    Raises ValueError if sz exceeds the array width/height after padding.
    '''

    # TODO implement
    # https://numpy.org/doc/1.18/reference/generated/numpy.pad.html will be helpful

    def op(sample: np.ndarray) -> np.ndarray:
        img = np.pad(sample, ((pad, pad), (pad, pad), (0, 0)), mode=pad_mode) if pad > 0 else sample

        w, h = sample.shape[0], sample.shape[1]
        th, tw = (sz, sz)
        if w < tw and h < th:
            raise ValueError('Crop section must not be greater than or equal to image.')

        i = np.random.randint(0, h + 2 * pad - sz)
        j = np.random.randint(0, w + 2 * pad - sz)

        assert j + tw <= img.shape[0]
        assert i + th <= img.shape[1]

        subimage = img[i:i + th, j:j + tw, :]

        return subimage

    return op


def scale_centered_crop(sz: int, probability: float) -> Op:
    '''
    Extract a square random crop of size sz from arrays with shape HWC.
    If pad is > 0, the array is first padded by pad pixels along the top, left, bottom, and right.
    How padding is done is governed by pad_mode, which should work exactly as the 'mode' argument of numpy.pad.
    Raises ValueError if sz exceeds the array width/height after padding.
    '''

    # https://numpy.org/doc/1.18/reference/generated/numpy.pad.html will be helpful

    def op(sample: np.ndarray) -> np.ndarray:

        if np.random.uniform() > (1 - probability):
            return sample

        w, h = sample.shape[0], sample.shape[1]
        th, tw = (sz, sz)
        if w < tw and h < th:
            raise ValueError('Crop section must not be greater than or equal to image.')

        i = int(np.floor((h - sz) / 2))
        j = int(np.floor((w - sz) / 2))

        subimage = sample[i:i + th, j:j + tw, :]

        return cv2.resize(subimage, dsize=(h, w), interpolation=cv2.INTER_CUBIC)

    return op


def rotate(max_angle: int) -> Op:
    def op(sample: np.ndarray) -> np.ndarray:
        angle = np.random.randint(-max_angle, max_angle)

        image_center = tuple(np.array(sample.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(sample, rot_mat, sample.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    return op


def debug(enabled=True, stop=False) -> Op:
    def op(sample: np.ndarray) -> np.ndarray:

        if not enabled:
            return sample

        x = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)

        plt.imshow(x)
        plt.show()

        if stop:
            print('Stopped')
        return sample

    return op

def resize(w,h) -> Op:
    def op(sample: np.ndarray) -> np.ndarray:
        if h != sample.shape[0] or w != sample.shape[1]:
            return cv2.resize(sample, (h,w))
            #im = np.zeros((h,w,3), dtype=sample.dtype)
            #im[0:32, 0:32, :] = sample
            #return im

        else:
            return sample
    
    return op