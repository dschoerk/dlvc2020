import os
import time
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
        self.fn = self.fn.astype(np.float32)
        self.fn /= (2**16-1)
        self.eps = eps

    def visualize(self) -> np.ndarray:
        '''
        Return a visualization as a color image. Use e.g. cv2.applyColorMap.
        Use the result to visualize the progress of gradient descent.
        '''
        #return self.fn.copy()
        return cv2.applyColorMap((self.fn * 255).astype(np.uint8), cv2.COLORMAP_JET)

    def __call__(self, loc: Vec2) -> float:
        '''
        Evaluate the function at location loc.
        Raises ValueError if loc is out of bounds.
        '''

        # TODO implement
        # You can simply round and map to integers. If so, make sure not to set eps and learning_rate too low
        # For bonus points you can implement some form of interpolation (linear should be sufficient)

        if loc.x1 < 0 or loc.x1 >= self.fn.shape[0] or loc.x2 < 0 or loc.x2 >= self.fn.shape[1]:
            raise ValueError("loc is out of bounds")

        # nearest neighbour
        x1_int = round(loc.x1)
        x2_int = round(loc.x2)
        return self.fn[x2_int, x1_int]


    def grad(self, loc: Vec2) -> Vec2:
        '''
        Compute the numerical gradient of the function at location loc, using the given epsilon.
        Raises ValueError if loc is out of bounds of fn or if eps <= 0.
        '''

        if self.eps <= 0:
            raise ValueError("eps should be > 0")

        if loc.x1 < 0 or loc.x1 >= self.fn.shape[0] or loc.x2 < 0 or loc.x2 >= self.fn.shape[1]:
            raise ValueError("loc is out of bounds")

        
        # we use numerical differentiation with central difference
        # TODO: check bounds
        d1 = self(Vec2(loc.x1 + self.eps, loc.x2)) - self(Vec2(loc.x1 - self.eps, loc.x2))
        d2 = self(Vec2(loc.x1, loc.x2 + self.eps)) - self(Vec2(loc.x1, loc.x2 - self.eps))
        print("%f %f" % (d1, d2))
        #exit(0)
        return Vec2(d1 / 2.0 / self.eps, d2 / 2.0 / self.eps)

if __name__ == '__main__':
    # Parse args
    import argparse

    parser = argparse.ArgumentParser(description='Perform gradient descent on a 2D function.')
    parser.add_argument('fpath', help='Path to a PNG file encoding the function')
    parser.add_argument('sx1', type=float, help='Initial value of the first argument')
    parser.add_argument('sx2', type=float, help='Initial value of the second argument')
    parser.add_argument('--eps', type=float, default=1.0, help='Epsilon for computing numeric gradients')
    parser.add_argument('--learning_rate', type=float, default=10.0, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0, help='Beta parameter of momentum (0 = no momentum)')
    parser.add_argument('--nesterov', action='store_true', help='Use Nesterov momentum')
    args = parser.parse_args()

    # Init
    fn = Fn(args.fpath, args.eps)
    vis = fn.visualize()
    loc = torch.tensor([args.sx1, args.sx2], requires_grad=True)
    last_loc = loc.detach().numpy()

    optimizer = torch.optim.SGD([loc], lr=args.learning_rate, momentum=args.beta, nesterov=args.nesterov)
    #optimizer = torch.optim.AdamW([loc], lr=args.learning_rate)

    # Perform gradient descent using a PyTorch optimizer
    # See https://pytorch.org/docs/stable/optim.html for how to use it
    while loc.grad is None or not np.isclose(Vec2(0, 0), loc.grad).all():
        # Visualize each iteration by drawing on vis using e.g. cv2.line()
        # Find a suitable termination condition and break out of loop once done

        optimizer.zero_grad()
        value = AutogradFn.apply(fn, loc)
        value.backward()       
        optimizer.step()

        loc_h = loc.detach().numpy()
        print(loc.grad)
        vis = cv2.line(vis, tuple(last_loc), tuple(loc_h), (0,0,255), 10)
        

        cv2.imshow('Progress', vis)
        cv2.waitKey(1)  # 20 fps, tune according to your liking

        print("dst sq %f" % np.linalg.norm(last_loc - loc_h))

        #if np.linalg.norm(last_loc - loc_h) < 1e-6:
        #    break

        last_loc = loc_h


# python optimizer_2d.py fn/camel6.png 250 500 --learning_rate 1000