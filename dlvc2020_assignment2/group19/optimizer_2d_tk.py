import os
from collections import namedtuple

import cv2
import torch
import numpy as np

Vec2 = namedtuple('Vec2', ['x1', 'x2'])


class AutogradFn(torch.autograd.Function):
    '''
    This class wraps a Fn instance to make it compatible with PyTorch optimimzers
    '''

    @staticmethod
    def forward(ctx, fn, loc):
        ctx.fn = fn
        ctx.save_for_backward(loc)
        value = fn(Vec2(loc[0].item(), loc[1].item()))
        return torch.tensor(value)

    @staticmethod
    def backward(ctx, grad_output):
        fn = ctx.fn
        loc, = ctx.saved_tensors
        grad = fn.grad(Vec2(loc[0].item(), loc[1].item()))
        return None, torch.tensor([grad.x1, grad.x2]) * grad_output


class Fn:
    '''
    A 2D function evaluated on a grid.
    '''

    def __init__(self, fpath: str, eps: float):
        '''
        Ctor that loads the function from a PNG file.
        Raises FileNotFoundError if the file does not exist.
        '''

        if not os.path.isfile(fpath):
            raise FileNotFoundError()

        self.fn = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        self.original = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        self.fn = self.fn.astype(np.float32)
        self.fn /= (2 ** 16 - 1)
        self.eps = eps

    def visualize(self, loc: Vec2) -> np.ndarray:
        '''
        Return a visualization as a color image. Use e.g. cv2.applyColorMap.
        Use the result to visualize the progress of gradient descent.
        '''
        color = (0, 0, 255)

        x1_pos = int(np.round(loc.x1))
        x2_pos = int(np.round(loc.x2))

        tmp = (self.original / 256).astype('uint8')
        color_image = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
        cv2.circle(color_image, (x1_pos, x2_pos), 1, color, 2)

        return color_image

    def round_value(self, x) -> (int, int):
        return (int(np.floor(x)), int(np.ceil(x)))

    def get_points(self, loc: Vec2):

        (x1_lower, x1_upper) = self.round_value(loc.x1)
        (x2_lower, x2_upper) = self.round_value(loc.x2)

        p1 = (x1_lower, x2_lower)
        p2 = (x1_lower, x2_upper)
        p3 = (x1_upper, x2_lower)
        p4 = (x1_upper, x2_upper)

        return [(p[0], p[1], self.fn[p[0]][p[1]]) for p in [p1, p2, p3, p4]]

    def bilinear_interpolation(self, x, y, points):
        '''Interpolate (x,y) from values associated with four points.
        https://stackoverflow.com/questions/8661537/how-to-perform-bilinear-interpolation-in-python
        The four points are a list of four triplets:  (x, y, value).
        The four points can be in any order.  They should form a rectangle.
        '''
        # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

        points = sorted(points)  # order points by x, then by y
        (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

        if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
            raise ValueError('points do not form a rectangle')
        if not x1 <= x <= x2 or not y1 <= y <= y2:
            raise ValueError('(x, y) not within the rectangle')

        return (q11 * (x2 - x) * (y2 - y) +
                q21 * (x - x1) * (y2 - y) +
                q12 * (x2 - x) * (y - y1) +
                q22 * (x - x1) * (y - y1)
                ) / ((x2 - x1) * (y2 - y1) + 0.0)

    def __call__(self, loc: Vec2) -> float:
        '''
        Evaluate the function at location loc.
        Raises ValueError if loc is out of bounds.
        '''

        points = self.get_points(loc)
        return self.bilinear_interpolation(loc.x1, loc.x2, points)

        # x1 = int(np.round(loc.x1))
        # x2 = int(np.round(loc.x2))
        # return self.fn[x1][x2]

        # You can simply round and map to integers. If so, make sure not to set eps and learning_rate too low
        # For bonus points you can implement some form of interpolation (linear should be sufficient)

    def grad(self, loc: Vec2) -> Vec2:
        '''
        Compute the numerical gradient of the function at location loc, using the given epsilon.
        Raises ValueError if loc is out of bounds of fn or if eps <= 0.
        '''

        if self.eps <= 0:
            raise ValueError('eps must not by less than 0')

        next_x1 = Vec2(loc.x1 + self.eps, loc.x2)
        prev_x1 = Vec2(loc.x1 - self.eps, loc.x2)

        next_x2 = Vec2(loc.x1, loc.x2 + self.eps)
        prev_x2 = Vec2(loc.x1, loc.x2 - self.eps)

        x_grad = (self(next_x1) - self(prev_x1)) / (2 * self.eps)
        y_grad = (self(next_x2) - self(prev_x2)) / (2 * self.eps)

        return Vec2(x_grad, y_grad)

if __name__ == '__main__':
    # Parse args
    import argparse

    # testinga()

    parser = argparse.ArgumentParser(description='Perform gradient descent on a 2D function.')
    parser.add_argument('fpath', help='Path to a PNG file encoding the function')
    parser.add_argument('sx1', type=float, help='Initial value of the first argument')
    parser.add_argument('sx2', type=float, help='Initial value of the second argument')
    parser.add_argument('--eps', type=float, default=0.0001, help='Epsilon for computing numeric gradients')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0, help='Beta parameter of momentum (0 = no momentum)')
    parser.add_argument('--nesterov', action='store_true', help='Use Nesterov momentum')
    args = parser.parse_args()

    # Init
    fn = Fn(args.fpath, args.eps)

    loc = torch.tensor([args.sx1, args.sx2], requires_grad=True)
    vis = fn.visualize(Vec2(loc.data[0], loc.data[1]))

    # cv2.imshow('Progress', vis)
    # while True:
    #     cv2.waitKey(50)  # 20 fps, tune according to your liking

    # optimizer = torch.optim.SGD([loc], lr=args.learning_rate, momentum=args.beta, nesterov=args.nesterov)
    optimizer = torch.optim.Adam([loc], lr=args.learning_rate)

    # Perform gradient descent using a PyTorch optimizer
    # See https://pytorch.org/docs/stable/optim.html for how to use it
    iteration = 0
    while loc.grad is None or not np.isclose(Vec2(0, 0), loc.grad).all():
        # Visualize each iteration by drawing on vis using e.g. cv2.line()
        # Find a suitable termination condition and break out of loop once done

        # clear the gradients
        optimizer.zero_grad()

        value = AutogradFn.apply(fn, loc)
        value.backward()

        optimizer.step()

        print(loc.grad)
        # print(loc)
        iteration += 1
        if iteration % 1000 == 0:
            # print(loc)
            vis = fn.visualize(Vec2(loc.data[0], loc.data[1]))
            cv2.imshow('Progress', vis)
            cv2.waitKey(5)  # 20 fps, tune according to your liking
    print('done')
    vis = fn.visualize(Vec2(loc.data[0], loc.data[1]))
    cv2.imshow('Progress', vis)
    while True:
        cv2.waitKey(5)  # 20 fps, tune according to your liking
