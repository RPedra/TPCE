import numpy as np
import matplotlib.pyplot as plt

def cost_function(array, roll_length):
    """Calculate the cost and waste for a given arrangement."""
    remained = roll_length
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
            roll_waste.append(roll_length - element)
            remained = roll_length - element
            total_cost += 1

    roll_waste[-1] = remained  # Update the waste for the last roll
    return total_cost, roll_contents, roll_waste

def generate_neighbor(solution):
    """Generate a neighboring solution by swapping two elements."""
    neighbor = np.copy(solution)
    i, j = np.random.randint(0, len(neighbor), size=2)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor

def simulated_annealing(requests, roll_length, initial_temp, cooling_rate, max_iter):
    """Solve the cutting stock problem using Simulated Annealing."""
    current_solution = np.copy(requests)
    np.random.shuffle(current_solution)
    current_cost, _, _ = cost_function(current_solution, roll_length)
    best_solution = np.copy(current_solution)
    best_cost = current_cost

    temperature = initial_temp

    for iteration in range(max_iter):
        neighbor = generate_neighbor(current_solution)
        neighbor_cost, _, _ = cost_function(neighbor, roll_length)

        # Accept the neighbor if it's better or probabilistically worse
        if neighbor_cost < current_cost or np.random.rand() < np.exp((current_cost - neighbor_cost) / temperature):
            current_solution = neighbor
            current_cost = neighbor_cost

            # Update the best solution found
            if current_cost < best_cost:
                best_solution = np.copy(current_solution)
                best_cost = current_cost

        # Cool down
        temperature *= cooling_rate

    _, roll_contents, roll_waste = cost_function(best_solution, roll_length)
    return best_cost, roll_contents, roll_waste

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

# Main execution
for i in range(1, 3):
    file_path = f'input{i}.stock'
    print(f"Test case {i}:")
    roll_length, requests = read_input_file(file_path)

    # Parameters for Simulated Annealing
    initial_temp = 1000
    cooling_rate = 0.95
    max_iter = 10000000

    best_cost, roll_contents, roll_waste = simulated_annealing(
        requests, roll_length, initial_temp, cooling_rate, max_iter
    )
    
    print(f"Best cost (number of rolls): {best_cost}")
    print(f"Total waste: {sum(roll_waste)}")
    plot_cutting_solution(roll_contents, roll_waste, roll_length)
