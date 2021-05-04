from random import randint

from Perceptron import PerceptronTrain, PerceptronTest
from linear import Vector, Scalar

SIZE = 500
TRAIN_SHARE = 0.9
TRAIN_SIZE = int(SIZE * TRAIN_SHARE)
EPOCHS = 1000

xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys = [Scalar(1) if (x.entries[0] * x.entries[1]).val <
      0 else Scalar(-1) for x in xs]

train_X, test_X = xs[:TRAIN_SIZE], xs[TRAIN_SIZE:]
train_Y, test_Y = ys[:TRAIN_SIZE], ys[TRAIN_SIZE:]

weights, bias = PerceptronTrain(list(zip(train_X, train_Y)), EPOCHS)

# TESTING

hits = 0
total = len(test_X)
for test_x, label in zip(test_X, test_Y):
    hits += PerceptronTest(weights, bias, test_x) == label.sign()

print(f"HITS: {hits}\nTOTAL: {total}\nHIT SHARE: {hits/total}")
breakpoint_anchor = 0
