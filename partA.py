import random
import numpy as np
import matplotlib.pyplot as plt

# (A)(i)

pop_size = 1000
pop = [''.join(random.choices(['0', '1'], k=30)) for _ in range(pop_size)]

print(pop)


# Fitness function
def ones_fitness(ind):
    return ind.count('1')


# Mutation function
def simple_mutate(ind):
    pos = random.randint(0, 29)
    ind = ind[:pos] + ('0' if ind[pos] == '1' else '1') + ind[pos + 1:]
    return ind


# Crossover function
def crossover(ind1, ind2):
    pos = random.randint(0, 29)
    ind1, ind2 = ind1[:pos] + ind2[pos:], ind2[:pos] + ind1[pos:]
    return ind1, ind2


# Main loop
def perform(fitness_func, population, mutation):
    generations = 100
    avg_fit = []
    for _ in range(generations):
        # Evaluate fitness
        fit = [fitness_func(ind) for ind in population]

        # Select parents using biased roulette wheel selection
        if sum(fit) > 0:
            probability = np.true_divide(fit, sum(fit))
            parents = np.random.choice(population, size=pop_size, replace=True, p=probability)
        else:
            parents = population

        # Generate offspring
        offspring = []
        for i in range(0, pop_size - 1, 2):
            ind1, ind2 = parents[i], parents[i + 1]
            ind1, ind2 = crossover(ind1, ind2)
            ind1, ind2 = mutation(ind1), mutation(ind2)
            offspring.extend([ind1, ind2])

        # Update population
        population = offspring

        # Calculate average fitness
        avg_fit.append(sum(fit) / len(fit))

    # Plot average fitness
    plt.plot(avg_fit)
    plt.xlabel('Generations')
    plt.ylabel('Average Fitness Using ' + fitness_func.__name__)
    plt.show()


perform(ones_fitness, pop, simple_mutate)


# (A)(ii)

def target_string_fitness(ind):
    target = '011100101011111111010110010101'
    return sum(1 for a, b in zip(ind, target) if a == b)


perform(target_string_fitness, pop, simple_mutate)


# (A)(iii)
def ones_fitness_iii(ind):
    if ind.count('1') > 0:
        return ind.count('1')
    else:
        return 2 * len(ind)


perform(ones_fitness_iii, pop, simple_mutate)


# (A)(iv)
def digit_target_fitness(ind):
    target = '077642659372610341864001850231'
    return sum(1 for a, b in zip(ind, target) if a == b)


def iv_mutate(ind):
    pos = random.randint(0, len(ind) - 1)
    possible_digits = list("0123456789")
    ind = ind[:pos] + random.choice(possible_digits) + ind[pos + 1:]
    return ind


perform(digit_target_fitness, pop, iv_mutate)


