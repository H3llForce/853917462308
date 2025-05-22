# Import statements
import time 
from tabulate import tabulate
from scipy import stats
from typing import List, Tuple, Optional
from tqdm import tqdm
import os
import sys
import math
import random
import numpy as np
import multiprocessing as mp
import traceback
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import csv

# Global configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Test configuration
DEFAULT_TEST_CITIES = 12
DEFAULT_SEED = 123
DEFAULT_TEST_RUNS = 100

# Visualization settings
import seaborn as sns
sns.set_style('whitegrid')
COLORS = {
    '2-opt': '#2ecc71',
    'PSO': '#3498db', 
    'ACO': '#e74c3c',
    'PSO+2-opt': '#9b59b6'
}

def save_to_csv(data, filename):
    """Save experiment results to CSV file"""
    if not data:
        print(f"No data to save for {filename}")
        return
    
    try:
        keys = data[0].keys()
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
        print(f"Saved results to {os.path.abspath(filename)}")
    except Exception as e:
        print(f"Error saving CSV: {str(e)}")
        traceback.print_exc()

def run_parallel_algorithm(args):
    """Wrapper for parallel execution with progress tracking"""
    alg_name, dist_matrix, run_idx = args
    try:
        result = run_algorithm(alg_name, dist_matrix, run_idx)
        return (alg_name, run_idx, *result)
    except Exception as e:
        print(f"Error in parallel run: {str(e)}")
        return (alg_name, run_idx, float('inf'), 0)

def plot_results(results, title): 
    """Plot comparative results from a single TSP instance run."""
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(12, 8))  # Increased figure size
    gs = GridSpec(2, 2, figure=fig)
    # Main title with optimal distance
    fig.suptitle(title, 
                fontsize=18, 
                y=0.97,
                fontweight='bold',
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.9, pad=8))

    # Plot distances with values
    ax1 = fig.add_subplot(gs[0, 0])
    distances = [v[0] for v in results.values()]
    algs = list(results.keys())
    bars = ax1.bar(algs, distances)
    
    # Add value labels to distance bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', 
                ha='center', va='bottom')
    
    ax1.set_title('Tour Distance', pad=15)
    ax1.set_ylabel('Distance')
    ax1.tick_params(axis='x', rotation=45)

    # Plot times with values
    ax2 = fig.add_subplot(gs[0, 1])
    times = [v[1] for v in results.values()]
    bars = ax2.bar(algs, times)
    
    # Add value labels to time bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', 
                ha='center', va='bottom')
    
    ax2.set_title('Execution Time', pad=15)
    ax2.set_ylabel('Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)

    # Plot normalized metrics
    ax3 = fig.add_subplot(gs[1, :])
    norm_dist = [v[2] for v in results.values()]
    norm_time = [v[3] for v in results.values()]
    
    x = np.arange(len(algs))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, norm_dist, width, label='Normalized Distance')
    bars2 = ax3.bar(x + width/2, norm_time, width, label='Normalized Time')
    
    for bars, vals in [(bars1, norm_dist), (bars2, norm_time)]:
        for i, (bar, val) in enumerate(zip(bars, vals)):
            bar.set_color(COLORS[algs[i]])
            ax3.text(bar.get_x() + bar.get_width()/2, val,
                    f'{val:.2f}', ha='center', va='bottom')
    
    ax3.set_title('Normalized Metrics', pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(algs, rotation=45)
    ax3.legend()
    ax3.grid(True)
    
    plt.subplots_adjust(
        top=0.88,   # 10% space at top for title
        bottom=0.1, # 8% space at bottom
        left=0.1,   # 8% space on left
        right=0.95,  # 5% space on right
        hspace=0.25,  # Space between rows
        wspace=0.15   # Space between columns
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92], pad=1.0, h_pad=1.0, w_pad=2.0)  # Increased padding
    return fig

def plot_test_results(test_results, optimal_distance=None):    
    """Plot statistical distribution of test run results."""
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(12, 8))  # Larger figure size
    gs = GridSpec(2, 2, figure=fig)
    fig.suptitle('Algorithm Performance Distribution',
                fontsize=20,
                y=0.96,
                fontweight='bold',
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.9, pad=10))

    # Box plot of distances
    ax1 = fig.add_subplot(gs[0, 0])
    distances = [test_results[alg]['distances'] for alg in test_results.keys()]
    bp = ax1.boxplot(distances, labels=test_results.keys(), patch_artist=True)
    
    for i, (patch, alg) in enumerate(zip(bp['boxes'], test_results.keys())):
        patch.set_facecolor(COLORS[alg])
    
    if optimal_distance:
        ax1.axhline(y=optimal_distance, color='r', linestyle='--', 
                    label='Optimal')
        ax1.legend()
    
    ax1.set_title('Distance Distribution')
    ax1.set_ylabel('Distance')
    ax1.tick_params(axis='x', rotation=45)

    # Box plot of execution times
    ax2 = fig.add_subplot(gs[0, 1])
    times = [test_results[alg]['runtimes'] for alg in test_results.keys()]
    bp = ax2.boxplot(times, labels=test_results.keys(), patch_artist=True)
    
    for i, (patch, alg) in enumerate(zip(bp['boxes'], test_results.keys())):
        patch.set_facecolor(COLORS[alg])
    
    ax2.set_title('Runtime Distribution')
    ax2.set_ylabel('Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)


    # Gap to optimal (if available)
    if optimal_distance:
        ax4 = fig.add_subplot(gs[1, 1])
        gaps = []
        
        for alg in test_results.keys():
            alg_gaps = [(d - optimal_distance) / optimal_distance * 100 
                       for d in test_results[alg]['distances']]
            gaps.append(alg_gaps)
        
        bp = ax4.boxplot(gaps, labels=test_results.keys(), patch_artist=True)
        
        for i, (patch, alg) in enumerate(zip(bp['boxes'], test_results.keys())):
            patch.set_facecolor(COLORS[alg])
        
        ax4.set_title('Gap to Optimal Solution')
        ax4.set_ylabel('Gap (%)')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.subplots_adjust(
        top=0.86,
        bottom=0.1,
        left=0.1,
        right=0.95,
        hspace=0.3,
        wspace=0.2
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92], pad=1.0, h_pad=1.0, w_pad=2.0)
    return fig

# Create a wrapper class to share data across processes
class SharedEnv:
    def __init__(self):
        self.matrix = None

shared_env = SharedEnv()

# Core algorithms and helper functions
def calculate_length(tour, matrix):
    """Calculate tour length"""
    return np.sum(matrix[np.roll(tour, 1), tour])

def calculate_swap_delta(tour, matrix, i, j):
    """Calculate change in tour length for a 2-opt swap"""
    n = len(tour)
    if i > j:
        i, j = j, i
    old_length = matrix[tour[i-1]][tour[i]] + matrix[tour[j]][tour[(j+1)%n]]
    new_length = matrix[tour[i-1]][tour[j]] + matrix[tour[i]][tour[(j+1)%n]]
    return new_length - old_length 

def two_opt(matrix, max_iter=500):
    """Enhanced 2-opt implementation"""
    n = matrix.shape[0]
    tour = list(range(n))
    length = calculate_length(tour, matrix)
    
    start_time = time.time()
    improved = True
    iteration = 0
    
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        
        for i in range(1, n-2):
            for j in range(i+1, n):
                delta = calculate_swap_delta(tour, matrix, i, j)
                if delta < 0:
                    tour[i:j+1] = reversed(tour[i:j+1])
                    length += delta
                    improved = True
                    break
            if improved:
                break
    
    runtime = time.time() - start_time
    return length, runtime, tour

# Brute force implementation
def brute_force_tsp(dist_matrix: np.ndarray, start_city: int = 0,
                   use_tqdm: bool = True) -> Tuple[float, List[int], float]:
    """Fast branch and bound algorithm for TSP"""
    n = dist_matrix.shape[0]
    other_cities = [i for i in range(n) if i != start_city]
    
    best_distance = float('inf')
    best_tour = None
    
    # Pre-compute minimum cost edges from each city
    min_edges = np.min(np.where(dist_matrix > 0, dist_matrix, float('inf')), axis=1)
    
    # Track search statistics
    nodes_processed = 0
    nodes_pruned = 0
    
    # Create a progress bar if requested
    pbar = None
    if use_tqdm:
        max_nodes = math.factorial(n)
        pbar = tqdm(total=max_nodes, desc="Branch and bound")
    
    def branch_and_bound(partial_tour: List[int], unvisited: List[int], 
                        current_dist: float, min_bound: float) -> None:
        nonlocal best_distance, best_tour, nodes_processed, nodes_pruned
        
        nodes_processed += 1
        if pbar and nodes_processed % 1000 == 0:
            pbar.update(1000)
        
        # Base case: all cities visited
        if not unvisited:
            final_dist = current_dist + dist_matrix[partial_tour[-1], start_city]
            if final_dist < best_distance:
                best_distance = final_dist
                best_tour = partial_tour.copy()
            return
        
        # Pruning: if lower bound exceeds best distance
        if current_dist + min_bound >= best_distance:
            nodes_pruned += 1
            return
        
        # Try each unvisited city
        last_city = partial_tour[-1]
        sorted_unvisited = sorted(unvisited, key=lambda city: dist_matrix[last_city, city])
        
        for next_city in sorted_unvisited:
            new_dist = current_dist + dist_matrix[last_city, next_city]
            if new_dist >= best_distance:
                continue
                
            new_unvisited = [c for c in unvisited if c != next_city]
            remaining_min_bound = sum(min_edges[city] for city in new_unvisited) / 2
            connection_to_start = min([dist_matrix[city, start_city] for city in new_unvisited]) if new_unvisited else 0
            new_bound = remaining_min_bound + connection_to_start
            
            partial_tour.append(next_city)
            branch_and_bound(partial_tour, new_unvisited, new_dist, new_bound)
            partial_tour.pop()
    
    # Initial bounds
    initial_min_bound = sum(min_edges[city] for city in other_cities) / 2
    connection_to_start = min([dist_matrix[city, start_city] for city in other_cities])
    initial_min_bound += connection_to_start
    
    # Start branch and bound
    start_time = time.time()
    branch_and_bound([start_city], other_cities, 0, initial_min_bound)
    runtime = time.time() - start_time
    
    # Close progress bar
    if pbar:
        pbar.close()
    
    # Print statistics
    print(f"Branch and bound statistics:")
    print(f"  Total nodes processed: {nodes_processed}")
    print(f"  Nodes pruned: {nodes_pruned} ({nodes_pruned/max(1, nodes_processed)*100:.2f}%)")
    print(f"  Time: {runtime:.4f} seconds")
    
    return best_distance, best_tour, runtime

def _solve_subproblem(args):
    """Helper function for parallel brute force"""
    dist_matrix, start_city, first_city = args
    
    # Local best solution
    local_best_distance = float('inf')
    local_best_tour = None
    
    # Cities remaining to visit after start_city and first_city
    n = dist_matrix.shape[0]
    other_cities = [i for i in range(n) if i != start_city]
    remaining_cities = [c for c in other_cities if c != first_city]
    
    # Initial partial tour
    partial_tour = [start_city, first_city]
    
    # Pre-compute the minimum cost edge from each city
    min_edges = np.min(np.where(dist_matrix > 0, dist_matrix, float('inf')), axis=1)
    
    def branch_and_bound(partial_tour, unvisited, current_dist, min_bound):
        nonlocal local_best_distance, local_best_tour
        
        # Base case: all cities visited
        if not unvisited:
            final_dist = current_dist + dist_matrix[partial_tour[-1], start_city]
            if final_dist < local_best_distance:
                local_best_distance = final_dist
                local_best_tour = partial_tour.copy()
            return
        
        # Pruning based on bound
        if current_dist + min_bound >= local_best_distance:
            return
        
        # Try each unvisited city
        last_city = partial_tour[-1]
        for next_city in sorted(unvisited, key=lambda city: dist_matrix[last_city, city]):
            new_dist = current_dist + dist_matrix[last_city, next_city]
            if new_dist >= local_best_distance:
                continue
            
            new_unvisited = [c for c in unvisited if c != next_city]
            remaining_min_bound = sum(min_edges[city] for city in new_unvisited) / 2
            connection_to_start = min([dist_matrix[city, start_city] for city in new_unvisited]) if new_unvisited else 0
            new_bound = remaining_min_bound + connection_to_start
            
            partial_tour.append(next_city)
            branch_and_bound(partial_tour, new_unvisited, new_dist, new_bound)
            partial_tour.pop()
    
    # Initial bounds
    initial_min_bound = sum(min_edges[city] for city in remaining_cities) / 2
    connection_to_start = min([dist_matrix[city, start_city] for city in remaining_cities]) if remaining_cities else 0
    initial_min_bound += connection_to_start
    
    # Calculate initial distance
    initial_dist = dist_matrix[start_city, first_city]
    
    # Start branch and bound for this subproblem
    branch_and_bound(partial_tour, remaining_cities, initial_dist, initial_min_bound)
    
    return local_best_tour, local_best_distance

def brute_force_tsp_parallel(dist_matrix: np.ndarray, start_city: int = 0,
                          num_jobs: int = -1) -> Tuple[float, List[int], float]:
    """Parallel implementation of brute force TSP"""
    if dist_matrix.shape[0] <= 10:
        return brute_force_tsp(dist_matrix, start_city)
    
    n = dist_matrix.shape[0]
    
    if num_jobs <= 0:
        num_jobs = mp.cpu_count()
    
    other_cities = [i for i in range(n) if i != start_city]
    args_list = [(dist_matrix.copy(), start_city, first_city) for first_city in other_cities]
    
    print(f"Starting parallel branch and bound with {min(num_jobs, len(other_cities))} workers...")
    
    start_time = time.time()
    with mp.Pool(processes=num_jobs) as pool:
        results = pool.map(_solve_subproblem, args_list)
    runtime = time.time() - start_time
    
    best_tour, best_distance = min(results, key=lambda x: x[1])
    
    print(f"Parallel branch and bound completed in {runtime:.4f} seconds")
    print(f"Best tour distance: {best_distance:.2f}")
    
    return best_distance, best_tour, runtime

def brute_force_tsp_auto(dist_matrix: np.ndarray, start_city: int = 0, 
                       use_tqdm: bool = True, num_jobs: int = -1) -> Tuple[float, List[int], float]:
    """Auto-select best brute force implementation"""
    n = dist_matrix.shape[0]
    if n <= 12:
        print(f"Using sequential branch and bound for {n}-city problem...")
        return brute_force_tsp(dist_matrix, start_city, use_tqdm)
    else:
        print(f"Using parallel branch and bound for {n}-city problem...")
        return brute_force_tsp_parallel(dist_matrix, start_city, num_jobs)

class PSO_TSP:
    """Pure PSO implementation for TSP"""    
    def __init__(self, matrix, num_particles=50, max_iter=100, timeout=60):
        self.matrix = matrix
        self.n = matrix.shape[0]
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.timeout = timeout
        self.particles = []
        self.g_best = None
        self.g_best_val = float('inf')
        
        # Initialize parameters
        self.w = 0.9  # Inertia weight
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        
        # Initialize particles
        self.initialize_diverse_particles()

    def _evaluate_position(self, particle):
        """Calculate fitness for a single particle position"""
        return calculate_length(particle['position'], self.matrix)

    def initialize_diverse_particles(self):
        """Initialize particles with some diversity"""
        # Random initialization
        for _ in range(self.num_particles // 2):
            tour = np.random.permutation(self.n)
            self._add_particle(tour)
            
        # Greedy nearest neighbor initialization
        for _ in range(self.num_particles - len(self.particles)):
            start = np.random.randint(self.n)
            tour = self._nearest_neighbor_tour(start)
            self._add_particle(tour)
    
    def _add_particle(self, tour):
        """Add a new particle"""
        fitness = calculate_length(tour, self.matrix)
        self.particles.append({
            'position': tour,
            'velocity': np.array([]),
            'best_position': tour.copy(),
            'best_fitness': fitness
        })
        
        if fitness < self.g_best_val:
            self.g_best = tour.copy()
            self.g_best_val = fitness
    
    def _nearest_neighbor_tour(self, start):
        """Simple nearest neighbor tour construction"""
        unvisited = set(range(self.n))
        tour = [start]
        unvisited.remove(start)
        
        while unvisited:
            current = tour[-1]
            next_city = min(unvisited, key=lambda x: self.matrix[current][x])
            tour.append(next_city)
            unvisited.remove(next_city)
        
        return np.array(tour)

    def update_particle(self, p):
        """Update particle position and velocity"""
        # Calculate segment sizes for position update
        i_size = max(2, int(self.n * self.w * 0.4))
        c_size = max(2, int(self.n * self.c1 * 0.3))
        s_size = max(2, int(self.n * self.c2 * 0.3))
        
        # Get segments from current position, personal best and global best
        inertia = self._get_segment(p['position'], i_size)
        cognitive = self._get_segment(p['best_position'], c_size)
        social = self._get_segment(self.g_best, s_size)
        
        # Build new position
        new_position = []
        used_cities = set()
        
        # Add segments in random order to maintain diversity
        segments = [inertia, cognitive, social]
        np.random.shuffle(segments)
        
        for segment in segments:
            for city in segment:
                if city not in used_cities:
                    new_position.append(city)
                    used_cities.add(city)
        
        # Add remaining cities using nearest neighbor heuristic
        missing = list(set(range(self.n)) - used_cities)
        if missing:
            current = new_position[-1] if new_position else np.random.choice(missing)
            while missing:
                next_city = min(missing, key=lambda x: self.matrix[current][x])
                new_position.append(next_city)
                missing.remove(next_city)
                current = next_city
        
        return np.array(new_position)
    
    def _get_segment(self, tour, size):
        """Get a segment of cities from a tour"""
        if len(tour) <= size:
            return tour
        
        # Select a random starting point
        start = np.random.randint(len(tour) - size + 1)
        return tour[start:start + size]
    
    def run(self):
        """Run PSO optimization"""
        start_time = time.time()
        last_improve_time = start_time
        prev_best = float('inf')
        
        for iter_num in range(self.max_iter):
            current_time = time.time()
            if current_time - start_time > self.timeout:
                break
            
            improved = False
            
            # Update all particles
            for p in self.particles:
                new_position = self.update_particle(p)
                new_fitness = calculate_length(new_position, self.matrix)
                
                # Update particle's best position
                if new_fitness < p['best_fitness']:
                    p['best_position'] = new_position.copy()
                    p['best_fitness'] = new_fitness
                    improved = True
                
                # Update particle's current position
                p['position'] = new_position
                
                # Update global best
                if new_fitness < self.g_best_val:
                    self.g_best = new_position.copy()
                    self.g_best_val = new_fitness
                    improved = True
            
            # Check improvement and early stopping
            if self.g_best_val < prev_best:
                prev_best = self.g_best_val
                last_improve_time = current_time
            elif current_time - last_improve_time > 5:  # No improvement for 5 seconds
                if not improved:
                    break
        
        total_time = time.time() - start_time
        return self.g_best_val, total_time

class HybridPSO2opt(PSO_TSP):
    """Hybrid PSO+2-opt implementation that applies 2-opt during PSO iterations"""
    def __init__(self, matrix, num_particles=50, max_iter=100, timeout=60, improvement_threshold=0.02):
        # Auto-scale particles based on CPU cores
        if num_particles is None:
            num_particles = max(50, mp.cpu_count() * 20)
            
        super().__init__(matrix, num_particles, max_iter, timeout)
        self.improvement_threshold = improvement_threshold

    def update_particle(self, p):
        """Update particle position and velocity (restore original implementation)"""
        # Remove the parallel evaluation code and keep original logic
        # Calculate segment sizes for position update
        i_size = max(2, int(self.n * self.w * 0.4))
        c_size = max(2, int(self.n * self.c1 * 0.3))
        s_size = max(2, int(self.n * self.c2 * 0.3))
        
        # Get segments from current position, personal best and global best
        inertia = self._get_segment(p['position'], i_size)
        cognitive = self._get_segment(p['best_position'], c_size)
        social = self._get_segment(self.g_best, s_size)
        
        # Build new position
        new_position = []
        used_cities = set()
        
        # Add segments in random order to maintain diversity
        segments = [inertia, cognitive, social]
        np.random.shuffle(segments)
        
        for segment in segments:
            for city in segment:
                if city not in used_cities:
                    new_position.append(city)
                    used_cities.add(city)
        
        # Add remaining cities using nearest neighbor heuristic
        missing = list(set(range(self.n)) - used_cities)
        if missing:
            current = new_position[-1] if new_position else np.random.choice(missing)
            while missing:
                next_city = min(missing, key=lambda x: self.matrix[current][x])
                new_position.append(next_city)
                missing.remove(next_city)
                current = next_city
        
        return np.array(new_position)
        
    def _apply_2opt_to_best_particles(self, top_k=3):
        """Apply 2-opt to the best k particles"""
        # Sort particles by fitness
        sorted_particles = sorted(self.particles, key=lambda x: x['best_fitness'])
        
        for p in sorted_particles[:top_k]:
            tour = p['position'].copy()
            improved = True
            best_delta = 0
            
            while improved:
                improved = False
                for i in range(1, self.n-2):
                    for j in range(i+1, self.n):
                        delta = calculate_swap_delta(tour, self.matrix, i, j)
                        if delta < best_delta:
                            best_delta = delta
                            best_i, best_j = i, j
                            improved = True
                
                if improved:
                    tour[best_i:best_j+1] = tour[best_i:best_j+1][::-1]
                    new_fitness = calculate_length(tour, self.matrix)
                    
                    # Update particle's position and best if improved
                    if new_fitness < p['best_fitness']:
                        p['position'] = tour.copy()
                        p['best_position'] = tour.copy()
                        p['best_fitness'] = new_fitness
                        
                        # Update global best if needed
                        if new_fitness < self.g_best_val:
                            self.g_best = tour.copy()
                            self.g_best_val = new_fitness
    
    def run(self):
        """Run hybrid PSO optimization with periodic 2-opt improvements"""
        start_time = time.time()
        last_improve_time = start_time
        prev_best = float('inf')
        
        for iter_num in range(self.max_iter):
            current_time = time.time()
            if current_time - start_time > self.timeout:
                break
            
            improved = False
            
            # Regular PSO updates
            for p in self.particles:
                new_position = self.update_particle(p)
                new_fitness = calculate_length(new_position, self.matrix)
                
                # Update particle's best position
                if new_fitness < p['best_fitness']:
                    p['best_position'] = new_position.copy()
                    p['best_fitness'] = new_fitness
                    improved = True
                
                # Update particle's current position
                p['position'] = new_position
                
                # Update global best
                if new_fitness < self.g_best_val:
                    self.g_best = new_position.copy()
                    self.g_best_val = new_fitness
                    improved = True
            
            # Apply 2-opt periodically
            if iter_num % 5 == 0:  # Every 5 iterations
                self._apply_2opt_to_best_particles()
              # Check improvement and early stopping
            if self.g_best_val < prev_best:
                # Calculate improvement safely
                if prev_best == float('inf'):
                    improvement = 1.0  # First improvement from infinity
                else:
                    improvement = (prev_best - self.g_best_val) / prev_best
                
                prev_best = self.g_best_val
                last_improve_time = current_time
                
                # Apply 2-opt more aggressively if close to convergence
                if improvement < self.improvement_threshold:
                    self._apply_2opt_to_best_particles(top_k=5)
            elif current_time - last_improve_time > 5:  # No improvement for 5 seconds
                if not improved:
                    break
        
        total_time = time.time() - start_time
        return self.g_best_val, total_time

class ACO_TSP:
    """Enhanced ACO implementation with local search"""
    def __init__(self, matrix, num_ants=15, max_iter=50, alpha=1, beta=2, rho=0.1):
        self.matrix = matrix
        self.n = matrix.shape[0]
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.pheromone = np.ones((self.n, self.n)) / self.n
        self.best_tour = None
        self.best_length = float('inf')
        self.last_improve_time = time.time()

    def run(self):
        start_time = time.time()
        stagnation_count = 0
        for iter in range(self.max_iter):
            # Construct ant tours
            ant_tours = self._construct_solutions()
            # Use the best ant's tour directly without local search
            best_tour = ant_tours[0][0]  # Best tour from current iteration
            best_length = ant_tours[0][1]
            # Update best solution
            if best_length < self.best_length:
                self.best_tour = best_tour.copy()
                self.best_length = best_length
                self.last_improve_time = time.time()
                stagnation_count = 0
            else:
                stagnation_count += 1
            # Early stopping conditions
            if stagnation_count > 10 or time.time() - self.last_improve_time > 5:
                break
            # Update pheromone using the best ant's tour
            self._update_pheromone([(best_tour, best_length)])
        return self.best_length, time.time() - start_time

    def _construct_solutions(self):
        ant_tours = []
        for _ in range(self.num_ants):
            tour = self._construct_tour()
            length = calculate_length(tour, self.matrix)
            ant_tours.append((tour, length))
        
        return sorted(ant_tours, key=lambda x: x[1])

    def _construct_tour(self):
        start = np.random.randint(self.n)
        tour = [start]
        visited = {start}
        
        while len(tour) < self.n:
            current = tour[-1]
            unvisited = list(set(range(self.n)) - visited)
            
            # Calculate probabilities
            pheromone = self.pheromone[current, unvisited]
            heuristic = 1.0 / (self.matrix[current, unvisited] + 1e-10)
            probabilities = (pheromone ** self.alpha) * (heuristic ** self.beta)
            probabilities = probabilities / probabilities.sum()
            
            # Select next city
            next_city = np.random.choice(unvisited, p=probabilities)
            tour.append(next_city)
            visited.add(next_city)
        
        return np.array(tour)

    def _local_search(self, tour):
        """Apply 2-opt local search"""
        improved = True
        while improved:
            improved = False
            for i in range(1, self.n-2):
                for j in range(i+1, self.n):
                    delta = calculate_swap_delta(tour, self.matrix, i, j)
                    if delta < 0:
                        tour[i:j+1] = tour[i:j+1][::-1]
                        improved = True
                        break
                if improved:
                    break
        return tour

    def _update_pheromone(self, ant_tours):
        # Evaporation
        self.pheromone *= (1 - self.rho)
        
        # Add new pheromone
        for tour, length in ant_tours[:10]:  # Only use top 10 ants
            value = 1.0 / length
            for i in range(len(tour)-1):
                self.pheromone[tour[i]][tour[i+1]] += value
                self.pheromone[tour[i+1]][tour[i]] += value  # Symmetric
        
        # Add extra pheromone to best tour
        if self.best_tour is not None:
            best_value = 1.0 / self.best_length
            for i in range(len(self.best_tour)-1):
                self.pheromone[self.best_tour[i]][self.best_tour[i+1]] += best_value * 2
                self.pheromone[self.best_tour[i+1]][self.best_tour[i]] += best_value * 2

def hybrid_pso_2opt(matrix, timeout=300):
    n = matrix.shape[0]
    
    # Initialize hybrid PSO with adaptive parameters
    num_particles = min(75, max(30, int(n * 0.6)))
    max_iter = min(300, max(100, n * 2))
    
    # Create and run hybrid PSO+2-opt
    hybrid = HybridPSO2opt(
        matrix=matrix,
        num_particles=num_particles,
        max_iter=max_iter,
        timeout=timeout,
        improvement_threshold=0.01
    )
    
    return hybrid.run()

def run_algorithm(alg_name, dist_matrix, run_idx):
    """Run a single algorithm test"""
    try:
        if alg_name == 'PSO':
            dist, runtime = PSO_TSP(dist_matrix, timeout=60).run()
        elif alg_name == 'ACO':
            dist, runtime = ACO_TSP(dist_matrix).run()
        elif alg_name == '2-opt':
            dist, runtime = two_opt(dist_matrix)[0:2]  # Only take first two return values
        else:  # PSO+2-opt
            dist, runtime = hybrid_pso_2opt(dist_matrix)
        return dist, runtime
    except Exception as e:
        print(f"Error running {alg_name} (run {run_idx}): {str(e)}")
        return float('inf'), 0

def run_algorithm_wrapped(args):
    """Wrapper for parallel execution"""
    alg_name, dist_matrix, run_idx = args
    return run_algorithm(alg_name, dist_matrix, run_idx)

def generate_test_cities(n_cities=DEFAULT_TEST_CITIES, seed=DEFAULT_SEED):
    """Generate random cities with fixed seed for testing"""
    random.seed(seed)
    np.random.seed(seed)
    coords = []
    for _ in range(n_cities):
        x = random.uniform(0, 1000)
        y = random.uniform(0, 1000)
        coords.append((x, y))
    return coords

def load_tsp(filename):
    """Load TSP file with error handling"""
    file_path = os.path.join(SCRIPT_DIR, f"{filename}.tsp")
    print(f"\nLoading TSP file: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f]
    except Exception as e:
        raise FileNotFoundError(f"File error: {str(e)}")

    coords = []
    read_coords = False
    
    for line in lines:
        if line.startswith("NODE_COORD_SECTION"):
            read_coords = True
            continue
        if line == "EOF":
            break
        if read_coords:
            parts = line.split()
            try:
                coords.append((float(parts[1]), float(parts[2])))
            except:
                continue

    if not coords:
        raise ValueError("No valid coordinates found!")
    
    print(f"Loaded {len(coords)} cities")
    return coords

def create_distance_matrix(coords):
    """Create distance matrix optimized for CPU execution"""
    n = len(coords)
    coords = np.array(coords)
    dx = coords[:, 0].reshape(n, 1) - coords[:, 0].reshape(1, n)
    dy = coords[:, 1].reshape(n, 1) - coords[:, 1].reshape(1, n)
    return np.sqrt(dx**2 + dy**2)

def run_experiment(matrix):
    """Run all algorithms sequentially with normalization"""
    results = {}
    
    algorithms = {
        '2-opt': lambda: two_opt(matrix)[0:2],
        'PSO': lambda: PSO_TSP(matrix, timeout=120).run(),
        'ACO': lambda: ACO_TSP(matrix).run(),
        'PSO+2-opt': lambda: hybrid_pso_2opt(matrix)
    }
    
    for alg_name, alg_func in algorithms.items():
        import time
        print(f"\nRunning {alg_name}...")
        alg_start = time.time()
        try:
            dist, runtime = alg_func()
            print(f"{alg_name} completed: Distance = {dist:.2f}, Time = {runtime:.2f}s")
            results[alg_name] = (dist, runtime)
        except Exception as e:
            print(f"{alg_name} failed: {str(e)}")
            traceback.print_exc()
            results[alg_name] = (float('inf'), float('inf'))
        finally:
            print(f"Total {alg_name} time: {time.time() - alg_start:.2f}s")
    
    # Find best values
    min_distance = min(v[0] for v in results.values())
    min_time = min(v[1] for v in results.values())
    
    # Add normalized values
    normalized_results = {}
    for alg, (dist, time) in results.items():
        norm_dist = dist / min_distance
        norm_time = time / min_time
        normalized_results[alg] = (dist, time, norm_dist, norm_time)
    
    return normalized_results
def run_test_mode(n_cities=DEFAULT_TEST_CITIES, seed=DEFAULT_SEED, n_runs=DEFAULT_TEST_RUNS):
    """Run algorithms multiple times on random test instance and collect statistics"""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate test instance
    print(f"\nGenerating random {n_cities}-city TSP instance with seed {seed}...")
    cities = np.random.rand(n_cities, 2) * 1000  # Cities in 1000x1000 square
    dist_matrix = create_distance_matrix(cities)
    
    # Calculate optimal solution if possible
    optimal_distance = None
    if n_cities <= 15:
        print("\nCalculating optimal solution...")
        try:
            opt_dist, _, _ = brute_force_tsp_auto(dist_matrix, use_tqdm=True)
            optimal_distance = opt_dist
            print(f"Optimal distance: {optimal_distance:.1f}")
        except Exception as e:
            print(f"Error calculating optimal solution: {str(e)}")
            traceback.print_exc()
    
    # Initialize results
    results_dict = {  # Renamed from results
        'PSO': {'distances': [], 'runtimes': []},
        'ACO': {'distances': [], 'runtimes': []},
        'PSO+2-opt': {'distances': [], 'runtimes': []},
        '2-opt': {'distances': [], 'runtimes': []}
    }
    
    csv_data = [] 

    # Run algorithms
    print(f"\nRunning {n_runs} tests for each algorithm...")
    total_tests = len(results_dict) * n_runs
    test_count = 0

    # Create parallel tasks
    tasks = []
    for alg in results_dict.keys():
        for i in range(n_runs):
            tasks.append((alg, dist_matrix, i))
    
    # Run in parallel using all cores
    print(f"\nRunning {len(tasks)} tests using {mp.cpu_count()} cores...")
    start_time = time.time()
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        parallel_results = list(tqdm(pool.imap_unordered(run_parallel_algorithm, tasks), 
                               total=len(tasks),
                               desc="Overall progress"))
    
    # Process results
    for alg_name, run_idx, dist, runtime in parallel_results:
        if dist != float('inf'):
            results_dict[alg_name]['distances'].append(dist)  # Changed variable name
            results_dict[alg_name]['runtimes'].append(runtime)
            csv_data.append({
                'mode': 'test',
                'problem': f'random_{n_cities}',
                'algorithm': alg_name,
                'run': run_idx+1,
                'distance': dist,
                'runtime': runtime,
                'optimal': optimal_distance
            })
    
    print(f"\nTotal execution time: {time.time()-start_time:.2f}s")
    
    # Calculate and print statistics
    for alg in results_dict.keys():
        distances = np.array(results_dict[alg]['distances'])
        times = np.array(results_dict[alg]['runtimes'])
        
        if len(distances) > 0:
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            mean_time = np.mean(times)
            std_time = np.std(times)
            min_dist = np.min(distances)
            
            print(f"\nStatistics for {alg}:")
            print(f"{'='*60}")
            print(f"Distance:")
            print(f"  Mean ± Std: {mean_dist:.2f} ± {std_dist:.2f}")
            print(f"  Best: {min_dist:.2f}")
            print(f"Runtime:")
            print(f"  Mean ± Std: {mean_time:.4f}s ± {std_time:.4f}s")
            
            if optimal_distance:
                gap = ((distances - optimal_distance) / optimal_distance * 100)
                print(f"Gap to Optimal:")
                print(f"  Mean ± Std: {np.mean(gap):.2f}% ± {np.std(gap):.2f}%")
                print(f"  Best: {np.min(gap):.2f}%")
            print(f"Success Rate: {len(distances)}/{n_runs}")
        else:
            print(f"\n{alg}: No successful runs")
    
    # Plot results
    try:
        plt.figure(figsize=(15, 10))
        fig = plot_test_results(results_dict, optimal_distance)
        plt.show()
    except Exception as e:
        print(f"\nError generating plots: {str(e)}")
        traceback.print_exc()
    
        if dist != float('inf'):
            results_dict[alg]['distances'].append(dist)
            results_dict[alg]['runtimes'].append(runtime)
            # Add this data collection
            csv_data.append({
                'mode': 'test',
                'problem': f'random_{n_cities}',
                'algorithm': alg,
                'run': i+1,
                'distance': dist,
                'runtime': runtime,
                'optimal': optimal_distance
            })
    
    # Save to CSV with full path
    save_to_csv(csv_data, os.path.join(SCRIPT_DIR, 'test_report.csv'))
    return results_dict, csv_data


def run_default_mode():
    """Default mode with predefined problems (single run)"""
    problems = ['berlin52', 'eil76', 'pr107', 'kroA100']
    optimal_distances = {
        'berlin52': 7542,
        'eil76': 538,
        'pr107': 44303,
        'kroA100': 21282,
    }
    
    csv_data = []
    
    for problem in problems:
        try:
            coords = load_tsp(problem)
            matrix = create_distance_matrix(coords)
            optimal = optimal_distances.get(problem)
            
            algorithms = {
                '2-opt': lambda: two_opt(matrix)[0:2],
                'PSO': lambda: PSO_TSP(matrix, timeout=120).run(),
                'ACO': lambda: ACO_TSP(matrix).run(),
                'PSO+2-opt': lambda: hybrid_pso_2opt(matrix)
            }
            
            for alg_name, alg_func in algorithms.items():
                print(f"\nRunning {alg_name} on {problem}...")
                dist, runtime = alg_func()
                csv_data.append({
                    'mode': 'default',
                    'problem': problem,
                    'algorithm': alg_name,
                    'run': 1,
                    'distance': dist,
                    'runtime': runtime,
                    'optimal': optimal
                })
        
        except Exception as e:
            print(f"Error processing {problem}: {str(e)}")
            traceback.print_exc()
    
    save_to_csv(csv_data, os.path.join(SCRIPT_DIR, 'default_report.csv'))
    return csv_data

def run_default_test_mode(n_runs=50):
    """Default test mode with multiple runs on predefined problems"""
    problems = ['berlin52', 'eil76', 'pr107', 'kroA100']
    optimal_distances = {
        'berlin52': 7542,
        'eil76': 538,
        'pr107': 44303,
        'kroA100': 21282,
    }
    
    csv_data = []
    
    for problem in problems:
        try:
            coords = load_tsp(problem)
            matrix = create_distance_matrix(coords)
            optimal = optimal_distances.get(problem)
            
            for run in range(n_runs):
                for alg_name in ['2-opt', 'PSO', 'ACO', 'PSO+2-opt']:
                    print(f"\nRun {run+1}/{n_runs} - {alg_name} on {problem}...")
                    dist, runtime = run_algorithm(alg_name, matrix, run)
                    csv_data.append({
                        'mode': 'default_test',
                        'problem': problem,
                        'algorithm': alg_name,
                        'run': run+1,
                        'distance': dist,
                        'runtime': runtime,
                        'optimal': optimal
                    })
        
        except Exception as e:
            print(f"Error processing {problem}: {str(e)}")
            traceback.print_exc()
    
    save_to_csv(csv_data, os.path.join(SCRIPT_DIR, 'default_test_report.csv'))
    return csv_data

def main():

    plt.ioff()
    
    print("\n" + "="*50)
    print("TSP Algorithm Benchmark Suite")
    print("="*50)
    print("Select mode:")
    print("1. Test Mode (Random cities, multiple runs)")
    print("2. Default Mode (Predefined problems, single run)")
    print("3. Default Test Mode (Predefined problems, multiple runs)")
    
    choice = input("Enter mode number (1-3): ").strip()
    
    if choice == '1':
        n_cities = int(input(f"Cities [Default: {DEFAULT_TEST_CITIES}]: ") or DEFAULT_TEST_CITIES)
        seed = int(input(f"Seed [Default: {DEFAULT_SEED}]: ") or DEFAULT_SEED)
        n_runs = int(input(f"Runs [Default: {DEFAULT_TEST_RUNS}]: ") or DEFAULT_TEST_RUNS)
        run_test_mode(n_cities, seed, n_runs)
        
    elif choice == '2':
        run_default_mode()
        
    elif choice == '3':
        runs = int(input(f"Number of runs [Default: 50]: ") or 50)
        run_default_test_mode(runs)
        
    else:
        print("Invalid choice")
    
    print("\nBenchmark complete. Reports saved to:")
    print("- test_report.csv (Test Mode)")
    print("- default_report.csv (Default Mode)")
    print("- default_test_report.csv (Default Test Mode)")

if __name__ == "__main__":
    # Configure for maximum CPU utilization
    mp.set_start_method('spawn', force=True)
    os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(mp.cpu_count())
    
    # Pin process to cores (Linux only)
    try:
        import psutil
        p = psutil.Process()
        p.cpu_affinity(list(range(mp.cpu_count())))
    except:
        pass
    
    main()
    print("\nPress any key to exit...")
    sys.stdin.read(1)
