import random

import numpy as np


class HillClimbingAlgorithm:
    def __init__(self, fitness_func, mutation_rate, step_rate, feature_size, maximize=True,
                 chromosome_size=64, generations=1000):
        self.fitness_func = fitness_func
        self.mutationRate = mutation_rate
        self.maximize = maximize
        self.chromosomeSize = chromosome_size
        self.stepRate = step_rate

        self.feature_size = feature_size

        self.generations = generations
        self.individual = []
        self.fitness = []
        self.samples = []

    def crossover(self, individuals1, individuals2):
        length = len(individuals1)
        new = [0] * length

        for ix, chromosome in enumerate(zip(individuals1, individuals2)):
            chromosome = list(chromosome)
            crossover_point = random.randrange(0, self.chromosomeSize - 1)
            mask = (1 << crossover_point) - 1
            new[ix] = chromosome[0] ^ ((chromosome[0] ^ chromosome[1]) & mask)

        return new

    def run(self):
        self.fitness_func(self.individual)
        self.samples.append(self.individual)
        self.fitness.append(self.fitness_func(self.individual))

        timeout = 0

        self.individual += [random.getrandbits(self.chromosomeSize) for _ in range(self.feature_size)]

        for generation in range(self.generations):
            new = self.individual.copy()

            for index in range(-len(new[(-self.feature_size * 15):]), 0):
                if np.random.choice([True, False], 1, p=[self.mutationRate, 1 - self.mutationRate]):
                    new[index] ^= 1 << random.randrange(0, self.chromosomeSize - 1)

                if np.random.choice([True, False], 1, p=[self.stepRate, 1 - self.stepRate]):
                    new[index] = abs(new[index] + random.choice([-1, 1]))

            f = self.fitness_func(new)
            if (f >= self.fitness[-1]) ^ (not self.maximize):
                self.individual = new
                self.samples.append(new)
                self.fitness.append(f)

                # print(f"Generation: {str(generation).zfill(6)}/{str(self.generations).zfill(6)} - Fitness: {f}")
                timeout = 0
            else:
                timeout += 1
                if timeout >= 35:
                    self.fitness.append(f)

                    last = self.individual[:7]
                    rnd1 = [random.getrandbits(self.chromosomeSize) for _ in range(self.feature_size)]
                    rnd2 = [random.getrandbits(self.chromosomeSize) for _ in range(self.feature_size)]
                    new = self.crossover(self.crossover(last, rnd1), rnd2)

                    self.individual += new
                    # self.individual += [random.getrandbits(self.chromosomeSize) for _ in range(self.feature_size)]
                    timeout = 0

            if (generation % int((self.generations / 1000)+0.5)) == 0:
                print(
                    f"Generation: {str(generation).zfill(6)}/{str(self.generations).zfill(6)} - Fitness: {self.fitness[-1]} - Size: {int(len(self.individual) / self.feature_size)}")
