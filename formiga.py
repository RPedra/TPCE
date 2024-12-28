import numpy as np
import matplotlib.pyplot as plt

def initialize_pheromone(max_request_value, initial_value=1.0):
    """Initialize pheromone levels for each possible item size."""
    return np.full(max_request_value + 1, initial_value)

def construct_solution(pheromones, requests, roll_length):
    """Construct a solution using probabilistic selection guided by pheromones."""
    solution = []
    remaining_requests = requests.copy()
    np.random.shuffle(remaining_requests)

    while remaining_requests.size > 0:
        current_roll = []
        remaining_space = roll_length

        while remaining_space > 0 and remaining_requests.size > 0:
            probabilities = pheromones[remaining_requests] / pheromones[remaining_requests].sum()
            chosen_index = np.random.choice(len(remaining_requests), p=probabilities)
            chosen_item = remaining_requests[chosen_index]

            if chosen_item <= remaining_space:
                current_roll.append(chosen_item)
                remaining_space -= chosen_item
                remaining_requests = np.delete(remaining_requests, chosen_index)
            else:
                break

        solution.append(current_roll)

    return solution

def calculate_solution_cost(solution, roll_length):
    """Calculate the cost of a solution and waste per roll."""
    total_cost = len(solution)
    roll_waste = [roll_length - sum(roll) for roll in solution]
    return total_cost, roll_waste

def update_pheromone(pheromones, solutions, costs, evaporation_rate):
    """Update pheromones based on solution quality."""
    pheromones *= (1 - evaporation_rate)
    for solution, cost in zip(solutions, costs):
        for roll in solution:
            for item in roll:
                pheromones[item] += 1 / cost
    return pheromones

def aco_solver(requests, roll_length, n_iterations, n_ants, evaporation_rate):
    """Ant Colony Optimization solver for the cutting stock problem."""
    max_request_value = max(requests)
    pheromones = initialize_pheromone(max_request_value)
    best_solution = None
    best_cost = float('inf')

    for iteration in range(n_iterations):
        solutions = []
        costs = []

        for _ in range(n_ants):
            solution = construct_solution(pheromones, requests, roll_length)
            cost, _ = calculate_solution_cost(solution, roll_length)
            solutions.append(solution)
            costs.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_solution = solution

        pheromones = update_pheromone(pheromones, solutions, costs, evaporation_rate)

    return best_cost, best_solution

def plot_cutting_solution(roll_contents, roll_waste, roll_length):
    """Plot the cutting solution."""
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

def read_input_file(file_path):
    """Reads the input file and extracts roll length and requests."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        roll_length = int(lines[0].split()[2])
        requests = np.array(list(map(int, lines[3].split(', '))), dtype=int)
    return roll_length, requests

def driver_ACO(file_path, n_iterations, n_ants, evaporation_rate):
    """Driver function to solve the cutting stock problem using ACO."""
    roll_length, requests = read_input_file(file_path)
    best_cost, best_solution = aco_solver(requests, roll_length, n_iterations, n_ants, evaporation_rate)
    roll_waste = [roll_length - sum(roll) for roll in best_solution]
    return best_cost, best_solution, roll_waste

# Example usage with multiple test cases
for i in range(1, 3):
    file_path = f'input{i}.stock'
    print(f"Test case {i}:")
    best_cost, best_solution, roll_waste = driver_ACO(file_path, n_iterations=100, n_ants=500, evaporation_rate=0.1)
    print(f"Best cost (number of rolls): {best_cost}")
    print(f"Total waste: {sum(roll_waste)}")
    roll_length, _ = read_input_file(file_path)
    plot_cutting_solution(best_solution, roll_waste, roll_length)
