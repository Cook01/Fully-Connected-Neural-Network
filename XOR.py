import math
import random
import numpy
from NeuralNetwork import NeuralNetwork



def tournament(population):
    best = None
    for individual in population:
        if best == None or individual[1] > best[1]:
            best = individual
    
    return best



#--------------------------------------------------------- Data ---------------------------------------------------------

data = [[[0, 0], 0], [[0, 1], 1], [[1, 0], 1], [[1, 1], 0]]

#--------------------------------------------------------- Initial Population ---------------------------------------------------------
nb_genetation = 10000

population_size = 2000
population = []

input = 2
hidden = [2]
output = 1

for i in range(population_size):
    population.append([NeuralNetwork(input, hidden, output), 0, 0])

#--------------------------------------------------------- Evaluate Fitness ---------------------------------------------------------

for generation in range(nb_genetation):
    print()
    print("---------------------------- Generation " + str(generation + 1) + " ----------------------------")

    for individual in population:

        for k in range(len(data)):
            result = individual[0].feedForward(data[k][0]).getA1()[0]

            error = numpy.absolute(data[k][1] - result)

            individual[1] += error

        
        individual[1] = len(data) - individual[1]
        individual[2] = (individual[1] / len(data)) * 100
        individual[1] = individual[1] ** 2

    
    next_population = []
    
    top = tournament(population)
    if top[2] >= 99.9:
        break

    next_population.append([top[0].copy(), 0, 0])

    print(top[2], "||", top[1])
    for _data in data:
        print(_data[0], "->", top[0].feedForward(_data[0]).getA1()[0])

    while len(next_population) < population_size / 2:
        parent1 = tournament(random.sample(population, int(population_size * 0.1)))
        parent2 = tournament(random.sample(population, int(population_size * 0.1)))

        next_population.append([parent1[0].copy(), 0, 0])
        next_population.append([parent2[0].copy(), 0, 0])
        next_population.append([parent1[0].crossover(parent2[0]), 0, 0])
        next_population.append([parent2[0].crossover(parent1[0]), 0, 0])

    #Fill remaining with new Neural Networks
    for i in range(population_size - len(next_population)):
        next_population.append([NeuralNetwork(input, hidden, output), 0, 0])
    
    population_size = len(next_population)
    population = next_population