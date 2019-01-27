# Adversarial-Attacks-in-Machine-Learning
A brief study on Adversarial Attacks and python scripts to generate and study them.

## Adversarial examples and attacks
Adversarial examples are morphed examples which lead a model into misclassifying them especially for relatively small changes in the input.
<br>
Various methods to generate them and defend them are studied here. Read this [paper](Adversarial-Attacks-in-Machine-Learning.pdf).

## Generating adversarial examples
A CNN model is trained on cifar10 in [model.py](model.py). Methods to generate adversarial examples are implemented in [adversarial_generator.py](adversarial_generator.py).
