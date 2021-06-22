import numpy as np
import math
import matplotlib.pyplot

# Chosen Data

A = 1000
Cp = 100
hr = 0.1


def additional_selling_price(b):
    pb = 30 * (1 - math.exp(-b / 1195))
    return pb


def additional_demand(p):
    dp = -0.003539 * p ** 3 + 2.1215 * p ** 2 - 413.3 * p + 26580
    return dp


# - - - - - - - - - - - Initialization - - - - - - - - - - -

def initialization(size):
    pop = []
    p = np.random.uniform(low=100, high=270, size=size)
    t = np.random.uniform(low=0, high=1, size=size)
    b = np.random.uniform(low=0, high=15000, size=size)
    for j in range(size):
        pop.append([p[j], t[j], b[j]])
    return pop


# - - - - - - - - - - - Fitness evaluation - - - - - - - - - - -

def cal_pop_fitness(pop):
    size = len(pop)
    fitness = []
    for j in range(size):
        pb = additional_selling_price(pop[j][2])
        dp = additional_demand(pop[j][0])
        tp = (pop[j][0] + pb) * dp - A / pop[j][1] - Cp * dp - hr * Cp * pop[j][1] * dp / 2 - pop[j][2]
        if tp <= 0:
            tp = 0
        fitness.append(tp)
    return fitness


# - - - - - - - - - - - Selection - - - - - - - - - - -

def pick_one(pop, fitness):
    size = len(pop)
    maxi = sum(fitness)
    pick = np.random.uniform(0, maxi)
    current = 0
    for k in range(size):
        current += fitness[k]
        if current > pick:
            return k


def selection(pop, fitness, rate):
    size = int(rate * len(pop))
    select = []
    for m in range(size):
        k = pick_one(pop, fitness)
        select.append(pop[k])
    return select


# - - - - - - - - - - - Crossover - - - - - - - - - - -

def cross_two(p1, p2):
    size = len(p1)
    half = size // 2
    c1 = p1[:half] + p2[half:]
    c2 = p2[:half] + p1[half:]
    return c1, c2


def crossover(pop, rate):
    size = len(pop)
    for n in range(size // 2):
        parents = np.random.randint(size, size=2)
        p1 = pop[parents[0]]
        p2 = pop[parents[1]]
        if np.random.uniform(0, 1) <= rate:
            c1, c2 = cross_two(p1, p2)
            pop[parents[0]] = c1
            pop[parents[1]] = c2
    return pop


# - - - - - - - - - - - Mutation - - - - - - - - - - -

def mutation(pop, rate):
    size = len(pop)
    for n in range(size):
        if np.random.uniform(0, 1) <= rate:
            p = np.random.uniform(100, 270)
            t = np.random.uniform(0, 1)
            b = np.random.uniform(0, 15000)
            pop[n] = [p, t, b]
    return pop


# - - - - - - - - - - - GA - - - - - - - - - - -

all = []

for i in range(100):

    pop_size = 100

    pop = initialization(pop_size)
    best_outputs = []

    for i in range(100):
        pop_fitness = cal_pop_fitness(pop)
        if best_outputs == [] or max(pop_fitness) > max(best_outputs):
            best_outputs.append(max(pop_fitness))
            ptb = pop[pop_fitness.index(max(pop_fitness))]
        else:
            best_outputs.append(max(best_outputs))
        pop = selection(pop, pop_fitness, 1)
        pop = crossover(pop, 0.2)
        pop = mutation(pop, 0.1)

    all.append(max(best_outputs))

print(all)
print(np.mean(all))
print(np.max(all))
print(np.min(all))
print(np.std(all))


# print(max(best_outputs))

# matplotlib.pyplot.plot(best_outputs)
# matplotlib.pyplot.xlabel("Iteration")
# matplotlib.pyplot.ylabel("Fitness")
# matplotlib.pyplot.show()
