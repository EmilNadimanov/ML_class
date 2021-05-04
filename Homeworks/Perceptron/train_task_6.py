import random

from linear import *
from matplotlib import pyplot
from Perceptron import PerceptronWeightGenerator, PerceptronTest, AveragedPerceptronTrainGenerator

from random import randint
random.seed("Deutschnofen".__hash__() % 2147483647)

SIZE = 500
TRAIN_SHARE = 0.9
TRAIN_SIZE = int(SIZE * TRAIN_SHARE)
EPOCHS = 100

v = Vector(randint(-100, 100), randint(-100, 100))
xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(SIZE)]
ys = [v * x * Scalar(randint(-1, 9)) for x in xs]

train_X, test_X = xs[:TRAIN_SIZE], xs[TRAIN_SIZE:]
train_Y, test_Y = ys[:TRAIN_SIZE], ys[TRAIN_SIZE:]
test_pairs = list(zip(test_X, test_Y))
total = len(test_X)  # For error rate


""" RANDOM PERMUTATION AT EACH EPOCH """
train_pairs = list(zip(train_X, train_Y))
"' First, the algorithm without caching '"
perceptronGenerator = PerceptronWeightGenerator(
    train_pairs,
    EPOCHS,
    permute_each_epoch=True
)
error_rates = list()  # FOR PLOTTING

for bias, weights in perceptronGenerator:
    hits = 0
    for test_x, label in test_pairs:
        hits += PerceptronTest(weights, bias, test_x) == label.sign()
    error_rates.append(1 - hits / total)

"' Second, the algorithm with caching '"
perceptronGenerator = AveragedPerceptronTrainGenerator(
    train_pairs,
    EPOCHS,
    permute_each_epoch=True
)
error_rates_cached = list()  # For plotting

for bias, weights in perceptronGenerator:
    hits = 0
    for test_x, label in test_pairs:
        hits += PerceptronTest(weights, bias, test_x) == label.sign()
    error_rates_cached.append(1 - hits / total)


pyplot.xlabel("Epochs")
pyplot.ylabel("Error rate")
pyplot.plot(
    range(EPOCHS),
    error_rates,
    'g-',
    label='Permute each epoch, no cache.')
pyplot.plot(
    range(EPOCHS),
    error_rates_cached,
    'r-',
    label='Permute each epoch, do cache.')
pyplot.legend()
pyplot.show()
