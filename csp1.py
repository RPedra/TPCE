import math
import random
import numpy as np
import matplotlib.pyplot as plt

def generate_permutation(array):
    '''This function generates a permutation of the given array'''
    permutation = np.copy(array)
    return np.random.permutation(permutation)

def cost_function(array, length):
    '''this function calculates number of rolls that 
    are used to fulfill the input requests and returns the roll contents and waste for each roll'''
    remained = length
    total_cost = 1
    roll_contents = [[]]
    roll_waste = [0]
    
    for element in array:
        if element <= remained:
            roll_contents[-1].append(element)
            remained -= element
        else:
            roll_waste[-1] = remained
            roll_contents.append([element])
            roll_waste.append(length - element)
            remained = length - element
            total_cost += 1
    
    roll_waste[-1] = remained  # Update the waste for the last roll
    return total_cost, roll_contents, roll_waste

def find_neighbor_HC(requests):
    '''This function finds neighbors for the current 
    permutation which is used in hill climbing algorithm'''
    neighbor = np.copy(requests)
    for i in range(2):
        ind1, ind2 = np.random.randint(len(requests)), np.random.randint(len(requests))
        neighbor[ind1], neighbor[ind2] = neighbor[ind2], neighbor[ind1]
    return neighbor

population_size = 100

def generate_population(chromosome):
    '''this function creates a generation with generation size length
    using the given chromosome'''
    chromosome_copy = np.copy(chromosome)
    population = np.array([])
    for _ in range(population_size):
        indivisual = np.random.permutation(chromosome_copy)
        population = np.append(population, indivisual)
        population = population.reshape(-1, len(indivisual))
    return population

def fitness(chromosome, length):
    '''this function calculates the fitness of the given permutation
    which is the number of rolls that are use. so the less the fitness, the better the result'''
    remained = length
    fitness_score = 1
    for gene in chromosome:
        if gene <= remained:
            remained -= gene
        else:
            remained = length - gene
            fitness_score += 1
    
    return fitness_score

def evaluate_population(population, length):
    '''this function calculates the fitness for every individual in the
    population'''
    population_fitness = np.array([])
    for indivisual in population:
        population_fitness = np.append(population_fitness, fitness(indivisual, length))
    return population_fitness

def tournament_selection(population, population_fitness, tournament_size=3):
    '''this function tries to find the best random parents for the new generation'''
    size = population[0].size
    parents = []
    parents_indices = []
    for i in range(population_size):
        candidates = random.sample(range(population_size), tournament_size)
        best_fit = np.inf
        parent_index = -1
        for c in candidates:
            if population_fitness[c] < best_fit:
                best_fit = population_fitness[c]
                parent_index = c
        parents.append(population[parent_index])
        parents_indices.append(parent_index)
    parents = np.array(parents)
    parents_indices = np.array(parents_indices)
    return parents, parents_indices

def crossover(population, parents, dictionary, requests, crossover_rate):
    '''this function tries to do the crossover with probability crossover rate'''
    children = []
    cnt = 0
    parents_copy = np.copy(parents)
    parents_copy = list(parents_copy)
    size = population[0].size

    while cnt < population_size:
        parent1, parent2 = parents_copy[cnt], parents_copy[cnt+1]
        child1, child2 = [], []

        cross_point = np.random.randint(1, size-2)

        for i in range(cross_point+1):
            child1.append(parent1[i])
            child2.append(parent2[i])
        
        child1_dict = {}
        child2_dict = {}

        for key in dictionary:
            child1_dict[key] = 0
            child2_dict[key] = 0

        for gene in child1:
            child1_dict[gene] += 1
        for gene in child2:
            child2_dict[gene] += 1

        choice = np.random.rand()

        if choice < crossover_rate:
            ind1, ind2 = cross_point+1, cross_point+1
            while len(child1) < len(parent1):
                if dictionary[parent2[ind1]] - child1_dict[parent2[ind1]] > 0:
                    child1_dict[parent2[ind1]] += 1
                    child1.append(parent2[ind1])
                ind1 = (1+ind1) % size

            while len(child2) < len(parent1):
                if dictionary[parent1[ind2]] - child2_dict[parent1[ind2]] > 0:
                    child2_dict[parent1[ind2]] += 1
                    child2.append(parent1[ind2])
                ind2 = (1+ind2) % size
        else:
            child1 = parent1[:]
            child2 = parent2[:]
        
        children.append(child1)
        children.append(child2)
        cnt += 2
    return np.array(children)

def generate_new_generation(population, parents_fitness, children, children_fitness, combine_rate):
    '''this function tries to make the new generation based on the previous
    population and the new children that are created'''
    new_generation = []
    population_copy = np.copy(population)
    population_copy = list(population_copy)
    for indivisual in population_copy:
        new_generation.append(indivisual)
    
    for i in range(int(combine_rate * population_size)):
        max_child = -1
        ind1 = -1
        for j in range(population_size):
            if children_fitness[j] < max_child:
                max_child = children_fitness[j]
                ind1 = j
        children_fitness[ind1] = np.inf

        min_parent = np.inf
        ind2 = -1
        for j in range(population_size):
            if parents_fitness[j] > min_parent:
                min_parent = parents_fitness[j]
                ind2 = j
        parents_fitness[ind2] = -1

        new_generation[ind2] = children[ind1]
    
    return new_generation

def mutation(children, mutation_rate):
    '''this function tries to do the mutation in the children'''
    for child in children:
        rate = np.random.rand()
        if rate < mutation_rate:
            ind1, ind2 = np.random.randint(len(child)), np.random.randint(len(child))
            child[ind1], child[ind2] = child[ind2], child[ind1]
    return children

def cutting_stock_GA_solver(length, chromosome, crossover_rate, mutation_rate, combine_rate, generation_length):
    '''this function tries to solve the cutting stocks problem with genetic algorithm'''
    population = generate_population(chromosome)
    population_fitness = evaluate_population(population, length)
    generation = 0
    best_fit = np.inf
    best_roll_contents = None
    best_roll_waste = None
    
    while generation < generation_length:
        parents, parents_indices_in_population = tournament_selection(population, population_fitness)
        parents_indices_in_population = np.array(parents_indices_in_population, dtype=int)

        parents_fitness = np.array([])
        for i in parents_indices_in_population:
            parents_fitness = np.append(parents_fitness, population_fitness[i])
        
        dictionary = {}
        for key in chromosome:
            if key not in dictionary:
                dictionary[key] = 1
            else:
                dictionary[key] += 1

        children = crossover(population, parents, dictionary, chromosome, crossover_rate)
        children = mutation(children, mutation_rate)
        children_fitness = evaluate_population(children, length)
        new_population = generate_new_generation(population, parents_fitness, children, children_fitness, combine_rate)
        new_population_fitness = evaluate_population(new_population, length)
        
        min_fitness = min(new_population_fitness)
        if min_fitness < best_fit:
            best_fit = int(min_fitness)
            best_index = np.argmin(new_population_fitness)
            # Correctly unpack three values
            best_fit, best_roll_contents, best_roll_waste = cost_function(new_population[best_index], length)
        
        population = np.array(new_population)
        generation += 1
    
    return best_fit, best_roll_contents, best_roll_waste

def driver_GA(file_path, crossover_rate, mutation_rate, combine_rate, generation_length):
    with open(file_path, 'r') as test_case:
        file_lines  = test_case.readlines()
        roll_length = int(file_lines[0].split()[2])
        requests    = np.array(list(map(int, file_lines[3].split(', ' ))), dtype=int)
        best_fit, roll_contents, roll_waste = cutting_stock_GA_solver(roll_length, requests, crossover_rate, mutation_rate, combine_rate, generation_length)
        return best_fit, roll_contents, roll_waste

def plot_cutting_solution(roll_contents, roll_waste, roll_length):
    fig, ax = plt.subplots()
    for i, roll in enumerate(roll_contents):
        current_position = 0
        for piece in roll:
            ax.broken_barh([(current_position, piece)], (i * 10, 9), facecolors=('tab:blue'))
            current_position += piece
        ax.broken_barh([(current_position, roll_waste[i])], (i * 10, 9), facecolors=('tab:red'))
    
    ax.set_ylim(0, len(roll_contents) * 10)
    ax.set_xlim(0, roll_length)
    ax.set_xlabel('Length')
    ax.set_ylabel('Rolls')
    ax.set_title('Cutting Stock Solution')
    plt.show()

for i in range(1, 3):
    file_p = 'input%i.stock' % (i)
    print("test case%i: " % (i))
    for crossover_rate in [0.5, 0.8]:
        for mutation_rate in [0.5, 0.8]:
            for combine_rate in [0.3, 0.8]:
                print(f"crossover rate: {crossover_rate:.1f}, mutation rate: {mutation_rate:.1f}, combination rate: {combine_rate:.1f}")
                best_fit, roll_contents, roll_waste = driver_GA(file_p, crossover_rate, mutation_rate, combine_rate, 200)
                print(f"Best fit (number of rolls): {best_fit}")
                #print("Roll contents:", roll_contents)
                #print("Roll waste:", roll_waste)
                
                # Plotting the cutting solution
                roll_length = int(open(file_p, 'r').readline().split()[2])
                plot_cutting_solution(roll_contents, roll_waste, roll_length)

                # Calculating and printing total waste
                total_waste = sum(roll_waste)
                print(f"Total waste: {total_waste}")
