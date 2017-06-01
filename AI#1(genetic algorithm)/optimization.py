import random
from math import *
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt1
from deap import base
from deap import creator
from deap import tools

def chromosomeDesign(individual):
    bit0, bit1, bit2, bit3, bit4, bit5, bit6, bit7 = individual[0:8]
    representation = (2**(-8)) * (bit7 * (2**7) + bit6 * (2**6) + bit5 * (2**5) + bit4 * (2**4) +
                                  bit3 * (2**3) + bit2 * (2**2) + bit1 * (2**1) + bit0 * (2**0)) *\
                      (30 - (-30)) + (-30)
    return representation

def showFitnessGraph(nthGeneration,avgList):
    fig = plt1.figure()
    fig.canvas.set_window_title('GA histogram')

    plt1.plot(range(nthGeneration),avgList,'ro')
    plt1.axis([-1,nthGeneration+1,0,2.5])
    plt1.title("average result of F(x,y)")
    plt1.xlabel("generation")
    plt1.ylabel("avgerage")
    plt1.show()
    fig.savefig("average.png", dpi=fig.dpi)

def showGraph(pop,nthGeneration):
    Xvalue,Yvalue,Fvalue = [],[],[]
    for i in range(len(pop)):
        Xvalue.append(-chromosomeDesign(pop[i][0:8]))
        Yvalue.append(-chromosomeDesign(pop[i][8:16]))
        Fvalue.append(-GrieWank(pop[i])[0], )

    fig = plt1.figure(figsize=(18,6),dpi=90)
    fig.suptitle(str(nthGeneration)+"th generation",fontsize="x-large")
    fig.canvas.set_window_title('GA histogram')

    plt1.subplot(1,3,1)
    plt1.hist2d(Xvalue, Fvalue, bins=50, norm=LogNorm())
    plt1.colorbar().ax.set_ylabel('Count')
    plt1.title("f(x,y) by x")
    plt1.xlabel("x")
    plt1.ylabel("f(x,y)")
    plt1.xlim([-30,30])
    plt1.ylim([0,2.4])

    plt1.subplot(1,3,2)
    plt1.hist2d(Yvalue, Fvalue, bins=50, norm=LogNorm())
    plt1.colorbar().ax.set_ylabel('Count')
    plt1.title("f(x,y) by y")
    plt1.xlabel("y")
    plt1.ylabel("f(x,y)")
    plt1.xlim([-30,30])
    plt1.ylim([0,2.4])

    plt1.subplot(1,3,3)
    plt1.hist2d(Xvalue, Yvalue, bins=50, norm=LogNorm())
    plt1.colorbar().ax.set_ylabel('Count')
    plt1.title("f(x,y) count")
    plt1.xlabel("x")
    plt1.ylabel("y")
    plt1.xlim([-30,30])
    plt1.ylim([-30,30])

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig(str(nthGeneration)+"th generation.png",dpi=fig.dpi)
    plt1.show()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, 16)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# the goal ('fitness') function to be maximized
def GrieWank(individual):
    x = chromosomeDesign(individual[0:8])
    y = chromosomeDesign(individual[8:16])
    return -(((x**2 + y**2)/4000.0) - (cos(x)*cos(y/sqrt(2)))+1),

# ----------
# Operator registration
# ----------
# register the goal / fitness function
toolbox.register("evaluate", GrieWank)

# register the crossover operator
toolbox.register("mate", tools.cxOnePoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(64)
    avgList = []
    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=2000)
    print(pop)
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    #
    # NGEN  is the number of generations for which the
    #       evolution runs
    CXPB, MUTPB, NGEN = 0.9, 0.1, 30

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))

    print(fitnesses)
    print(len(fitnesses))

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Begin the evolution
    for g in range(NGEN):
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # print(offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

        avgList.append(-mean)
        if(g%10 == 0):
            showGraph(pop=pop, nthGeneration=g)

    print("-- End of (successful) evolution --")

    print(pop)

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    print("(%s, %s)" % (chromosomeDesign(best_ind[0:8]), chromosomeDesign(best_ind[8:16])))

    showGraph(pop=pop, nthGeneration=NGEN)
    showFitnessGraph(nthGeneration=NGEN, avgList=avgList)

if __name__ == "__main__":
    main()