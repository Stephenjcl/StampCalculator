# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:21:12 2019

@author: Stephen
"""

# Inputs of the equation.
equation_inputs = [5,92,120,127,180,190]
postage = 444


import numpy

def cal_pop_fitness(equation_inputs, pop, target):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    pop = pop.clip(min=0)
    fitness = numpy.sum(pop*equation_inputs, axis=1)
    fitness = fitness - target
    #fitness = abs(fitness)
    #fitness = fitness * -1
    #fitness = 1/(abs(pop*equation_inputs))
    return fitness

# def find_nearest(array, value):
#     array = numpy.asarray(array)
#     idx = (numpy.abs(array - value)).argmin()
#     return array[idx]

def find_nearest_above(my_array, target):
    diff = my_array - target + 1
    mask = numpy.ma.less_equal(diff, 0)
    # We need to mask the negative differences and zero
    # since we are looking for values above
    if numpy.all(mask):
        return numpy.where(my_array == numpy.max(my_array)) #Returns the highest fitness value if there is no value above target.
    masked_diff = numpy.ma.masked_array(diff, mask)
    return masked_diff.argmin()

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        #max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        #max_fitness_idx = max_fitness_idx[0][0]
        
        max_fitness_idx = find_nearest_above(fitness, 0)
        #max_fitness_idx = max_fitness_idx[0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -999999999
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually it is at the center.
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        random_value = numpy.random.randint(-1, 1, 1)
        gene_target = numpy.random.randint(0,num_weights)
        offspring_crossover[idx, gene_target] = offspring_crossover[idx, gene_target] + random_value
    return offspring_crossover

"""
The y=target is to maximize this equation ASAP:
    y = w1x1+w2x2+w3x3
    where (x1,x2,x3)=(90,127,190)
    What are the best values for the 3 weights w1 to w3?
    We are going to use the genetic algorithm for the best possible values after a number of generations.
"""



# Number of the weights we are looking to optimize.
num_weights = len(equation_inputs)

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = len(equation_inputs)*10
num_parents_mating = int(sol_per_pop/2)

# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.

"""
Here I have set the total iterations to 12. This means 12 repetitions of 10 generations.
Since some iterations will converge to the wrong local minima, I decided just starting
over a few more times was easier. The best answer will be pulled at the end of the 12
runs.
"""
replicates = 12

resultfit = [0]*replicates
resultstamps = [0]*replicates

for runnum in range(0,replicates):
    #Creating the initial population.
    new_population = numpy.random.uniform(low=0, high=4.0, size=pop_size)
    new_population = new_population.astype(int)
    new_population = new_population.clip(min=0)
    #print(new_population)
    
    num_generations = 10
    for generation in range(num_generations):
        print("Generation : ", generation)
        # Measing the fitness of each chromosome in the population.
        fitness = cal_pop_fitness(equation_inputs, new_population, postage)
    
        # Selecting the best parents in the population for mating.
        parents = select_mating_pool(new_population, fitness, 
                                          num_parents_mating)
    
        # Generating next generation using crossover.
        offspring_crossover = crossover(parents,
                                           offspring_size=(pop_size[0]-parents.shape[0], num_weights))
    
        # Adding some variations to the offsrping using mutation.
        offspring_mutation = mutation(offspring_crossover)
    
        # Creating the new population based on the parents and offspring.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation
        new_population = new_population.clip(min=0)
    
        # The best result in the current iteration.
        print("Best result : ", numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))
    
    # Getting the best solution after iterating finishing all generations.
    #At first, the fitness is calculated for each solution in the final generation.
    fitness = cal_pop_fitness(equation_inputs, new_population, postage)
    # Then return the index of that solution corresponding to the best fitness.
    #best_match_idx = numpy.where(fitness == numpy.max(fitness))
    best_match_idx = find_nearest_above(fitness, 0)
    
    print("Best solution : ", new_population[best_match_idx, :])
    print("Best solution fitness : ", fitness[best_match_idx])
    
    resultfit[runnum] = fitness[best_match_idx]
    resultstamps[runnum] = new_population[best_match_idx, :]
    
resultfit = numpy.asarray(resultfit)
resultstamps = numpy.asarray(resultstamps)

best_match_idx = find_nearest_above(resultfit, 0)
print("Final Solution : ", resultstamps[best_match_idx])
print("Final Fitness : ", resultfit[best_match_idx])

#Below here we print an "English legible" output that describes the exact stamps you need.
stamps_printout = resultstamps[best_match_idx]
print("You require : ")
for i in range(0, len(stamps_printout)):
    print(stamps_printout[i], " - ", equation_inputs[i], "¢ stamp")
print("to obtain ", numpy.sum(stamps_printout*equation_inputs), "¢ in total")