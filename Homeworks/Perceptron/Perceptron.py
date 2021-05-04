"""
Algorithm 5 PerceptronTrain(D, MaxIter)
1: w_d ← 0, for all d = 1 . . . D // initialize weights
2: b ← 0 // initialize bias
3: for iter = 1 . . . MaxIter do
    4: for all (x,y) ∈ D do
        5: a ← ∑^D_d=1 (w_d * x_d) + b // compute activation for this example
        6: if ya ≤ 0 then
            7: w_d ← w_d + yxd, for all d = 1 . . . D // update weights
            8: b ← b + y // update bias
        9: end if
    10: end for
11: end for
12: return w0, w1, . . . , wD, b
"""
from typing import Tuple, List
from linear import Vector, Scalar
import random
from random import shuffle
random.seed("Colmar Tropicale".__hash__() % 2147483647)


def PerceptronTrain(D: List[Tuple[Vector, Scalar]],
                    MaxIter: int):
    if not all(map(lambda x: len(x) == len(D[0][0]), D)):
        raise Exception("Not all vectors in matrix D are of equal length")
    # if all vectors are equal in size, we can take one of them to compute
    # weights' magnitude
    weights = Vector.zero(D[0][0].magnitude().val)
    bias = Scalar(0)
    for epoch in range(MaxIter):
        for X, Y in D:
            activation: Scalar = X * weights + bias
            if (Y * activation).val <= 0:
                weights = weights + Y * X
                bias += Y
    return weights, bias


"""
Algorithm 6 PerceptronTest(w0, w1, . . . , wD, b, xˆ)
1: a ← ∑^D_d=1 (w_d * ^x_d) + b // compute activation for the test example
2: return sign(a)
"""


def PerceptronTest(weights: Vector, bias: Scalar, X_bar: Vector):
    activation = X_bar * weights + bias
    return activation.sign()


def PerceptronWeightGenerator(
        D: List[Tuple[Vector, Scalar]],
        MaxIter: int,
        permute_each_epoch=False):
    if not all(map(lambda x: len(x) == len(D[0][0]), D)):
        raise Exception("Not all vectors in matrix D are of equal length")
    # if all vectors are equal in size, we can take one of them to compute
    # weights' magnitude
    weights = Vector.zero(D[0][0].magnitude().val)
    bias = Scalar(0)
    ##
    for epoch in range(MaxIter):
        for idx in range(len(D)):
            if permute_each_epoch is True:
                # shuffle the remaining examples
                D[idx:] = random.sample(D[idx:], len(D[idx:]))
            X, Y = D[idx]
            activation: Scalar = X * weights + bias
            if (Y * activation).val <= 0:
                weights = weights + (Y * X)
                bias += Y
        yield bias, weights


"""
Algorithm 7 AveragedPerceptronTrain(D, MaxIter)
1: w ← {0, 0, . . . 0i , b ← 0 // initialize weights and bias
2: u ← h0, 0, . . . 0i , β ← 0 // initialize cached weights and bias
3: c ← 1 // initialize example counter to one
4: for iter = 1 . . . MaxIter do
    5: for all (x,y) ∈ D do
        6: if y(w · x + b) ≤ 0 then
            7: w ← w + y x // update weights
            8: b ← b + y // update bias
            9: u ← u + y c x // update cached weights
            10: β ← β + y c // update cached bias
        11: end if
        12: c ← c + 1 // increment counter regardless of update
    13: end for
14: end for
15: return (w - 1/c*u), (b - 1/c*β) // return averaged weights and bias
"""


def AveragedPerceptronTrain(D: List[Tuple[Vector, Scalar]],
                            MaxIter: int,
                            permute_each_epoch=False):
    weights = Vector.zero(D[0][0].magnitude().val)
    bias = Scalar(0)

    cached_weights = Vector.zero(D[0][0].magnitude().val)
    beta = Scalar(0)
    c = Scalar(1)

    for epoch in range(MaxIter):
        for idx in range(len(D)):
            if permute_each_epoch is True:
                # shuffle the remaining examples
                D[idx:] = random.sample(D[idx:], len(D[idx:]))
            X, Y = D[idx]
            activation: Scalar = X * weights + bias
            if (Y * activation).val <= 0:
                weights = weights + Y * X
                bias += Y
                cached_weights = cached_weights + Y * c * X
                beta += Y * c
            c.val += 1
    return bias - beta / c, weights - cached_weights / c


def AveragedPerceptronTrainGenerator(D: List[Tuple[Vector, Scalar]],
                                     MaxIter: int,
                                     permute_each_epoch=False):
    weights = Vector.zero(D[0][0].magnitude().val)
    bias = Scalar(0)

    cached_weights = Vector.zero(D[0][0].magnitude().val)
    beta = Scalar(0)
    c = Scalar(1)

    for epoch in range(MaxIter):
        for idx in range(len(D)):
            if permute_each_epoch is True:
                # shuffle the remaining examples
                D[idx:] = random.sample(D[idx:], len(D[idx:]))
            X, Y = D[idx]
            activation: Scalar = X * weights + bias
            if (Y * activation).val <= 0:
                weights = weights + Y * X
                bias += Y
                cached_weights = cached_weights + Y * c * X
                beta += Y * c
            c.val += 1
        yield bias - beta / c, weights - cached_weights / c
