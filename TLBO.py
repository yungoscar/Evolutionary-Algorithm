import numpy
import numpy as np
import math
import matplotlib.pyplot


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
        fitness.append(tp)
    return fitness

# - - - - - - - - - - - Teacher - - - - - - - - - - -

def Teacher_phase(pop,pop_fitness):
    for i in range (len(pop)):
        if (pop_fitness[i] == max(pop_fitness)):
            indice = i
    mean = numpy.mean(pop,0)
    r1 = np.random.uniform(0,1)
    r2 = np.random.uniform(0, 1)
    r3 = np.random.uniform(0, 1)
    Tf = 1
    DM1 = r1 * Tf * (pop[indice][0] - mean[0])
    DM2 = r2 * Tf * (pop[indice][1] - mean[1])
    DM3 = r3 * Tf * (pop[indice][2] - mean[2])
    pop_learners = []
    for i in range (len(pop)):
        A = pop[i][0] + DM1
        B = pop[i][1] + DM2
        C = pop[i][2] + DM3
        pop_learners.append([A,B,C])
    for i in range (len(pop)):
        if (pop_learners[i][0] < 100):
            pop_learners[i][0] = 100
        if (pop_learners[i][0] > 270):
            pop_learners[i][0] = 270
        if (pop_learners[i][1] < 0):
            pop_learners[i][1] = 0.0000000001
        if (pop_learners[i][1] > 1):
            pop_learners[i][1] = 1
        if (pop_learners[i][2] < 0):
            pop_learners[i][2] = 0
        if (pop_learners[i][2] > 15000):
            pop_learners[i][2] = 15000
    learners_fitness = cal_pop_fitness(pop_learners)
    for i in range (len(pop)):
        if (learners_fitness[i] > pop_fitness[i]):
            pop[i][0] = pop_learners[i][0]
            pop[i][1] = pop_learners[i][1]
            pop[i][2] = pop_learners[i][2]
    return pop

# - - - - - - - - - - - Learner - - - - - - - - - - -

def Learner_phase(pop,pop_fitness):
    pop_learners = np.copy(pop)
    for i in range (len(pop)):
        choose = i
        while (choose == i):
            choose = np.random.randint(0,len(pop))
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)
        r3 = np.random.uniform(0, 1)
        if (pop_fitness [choose] > pop_fitness[i]):
            X1 = pop[i][0] + r1 * (pop[choose][0] - pop[i][0])
            X2 = pop[i][1] + r2 * (pop[choose][1] - pop[i][1])
            X3 = pop[i][2] + r3 * (pop[choose][2] - pop[i][2])
            pop_learners[i][0] = X1
            pop_learners[i][1] = X2
            pop_learners[i][2] = X3
        else:
            X1 = pop[choose][0] + r1 * (pop[i][0] - pop[choose][0])
            X2 = pop[choose][1] + r2 * (pop[i][1] - pop[choose][1])
            X3 = pop[choose][2] + r3 * (pop[i][2] - pop[choose][2])
            pop_learners[choose][0] = X1
            pop_learners[choose][1] = X2
            pop_learners[choose][2] = X3

        if (pop_learners[i][0] < 100):
            pop_learners[i][0] = 100
        if (pop_learners[i][0] > 270):
            pop_learners[i][0] = 270
        if (pop_learners[i][1] < 0):
            pop_learners[i][1] = 0
        if (pop_learners[i][1] > 1):
            pop_learners[i][1] = 1
        if (pop_learners[i][2] < 0):
            pop_learners[i][2] = 0
        if (pop_learners[i][2] > 15000):
            pop_learners[i][2] = 15000
    learners_fitness = cal_pop_fitness(pop_learners)
    for i in range (len(pop)):
        if (learners_fitness[i] > pop_fitness[i]):
            pop[i][0] = pop_learners[i][0]
            pop[i][1] = pop_learners[i][1]
            pop[i][2] = pop_learners[i][2]
    return pop





best_outputs = []
pop_size = 100
pop = initialization(pop_size)
pop_fitness = cal_pop_fitness(pop)
# print (pop)
# print (max(pop_fitness))
# print (pop_fitness)
# Teacher_phase(pop,pop_fitness)
# pop_fitness = cal_pop_fitness(pop)
# print (pop_fitness)
# Learner_phase(pop,pop_fitness)
# pop_fitness = cal_pop_fitness(pop)
# print(pop_fitness)

for i in range (20):
    Teacher_phase(pop, pop_fitness)
    pop_fitness = cal_pop_fitness(pop)
    Learner_phase(pop, pop_fitness)
    pop_fitness = cal_pop_fitness(pop)
    best_outputs.append(max(pop_fitness))
print (pop_fitness)
print (max(pop_fitness))

print(max(best_outputs))

matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()
