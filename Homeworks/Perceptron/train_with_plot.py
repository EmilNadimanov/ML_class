import random

from linear import *
from matplotlib import pyplot
from Perceptron import PerceptronWeightGenerator, PerceptronTest

from random import randint, shuffle
random.seed("Peaches and cream".__hash__() % 2147483647)

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

""" NO PERMUTATION WHATSOEVER """
train_pairs = list(zip(train_X, train_Y))
perceptronGenerator = PerceptronWeightGenerator(
    train_pairs,
    EPOCHS
)
error_rates_original = list()  # FOR PLOTTING
total = len(test_X)  # For error rate

for bias, weights in perceptronGenerator:
    hits = 0
    for test_x, label in test_pairs:
        hits += PerceptronTest(weights, bias, test_x) == label.sign()
    error_rates_original.append(1 - hits / total)

""" RANDOM PERMUTATION AT THE BEGINNING """
train_pairs = list(zip(train_X, train_Y))
shuffle(train_pairs)
perceptronGenerator = PerceptronWeightGenerator(
    train_pairs,
    EPOCHS
)
error_rates_perm_once = list()  # FOR PLOTTING

for bias, weights in perceptronGenerator:
    hits = 0
    for test_x, label in test_pairs:
        hits += PerceptronTest(weights, bias, test_x) == label.sign()
    error_rates_perm_once.append(1 - hits / total)

""" RANDOM PERMUTATION AT EACH EPOCH """
train_pairs = list(zip(train_X, train_Y))
perceptronGenerator = PerceptronWeightGenerator(
    train_pairs,
    EPOCHS,
    permute_each_epoch=True
)
error_rates_perm_each_epoch = list()  # FOR PLOTTING

for bias, weights in perceptronGenerator:
    hits = 0
    for test_x, label in test_pairs:
        hits += PerceptronTest(weights, bias, test_x) == label.sign()
    error_rates_perm_each_epoch.append(1 - hits / total)

pyplot.xlabel("Epochs")
pyplot.ylabel("Error rate")
pyplot.plot(range(EPOCHS), error_rates_original, 'r-', label='No permutation.')
pyplot.plot(range(EPOCHS), error_rates_perm_once, 'b-',
            label='Permute once at the very beginning.')
pyplot.plot(
    range(EPOCHS),
    error_rates_perm_each_epoch,
    'g-',
    label='Permute each epoch.')
pyplot.legend()
pyplot.show()
