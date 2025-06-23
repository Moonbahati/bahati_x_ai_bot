import random
import multiprocessing
import logging
from copy import deepcopy

# Ultra Modules
from core.digit_predictor.ensemble_voter import evaluate_strategy
from engine.dna_profiler import check_dna_uniqueness
from ai.fraud_detection_ai import detect_fraud_pattern
from ai.auto_feedback_loop import feedback_evaluator

# Logger Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegendaryUltraGeneticOptimizer")

class GeneticOptimizer:
    def __init__(self, population_size=100, generations=50, crossover_rate=0.9, 
                 mutation_rate=0.02, elitism=True, genome_length=10, bounds=(0, 1),
                 parallel=False, threads=4):
        
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.genome_length = genome_length
        self.bounds = bounds
        self.parallel = parallel
        self.threads = threads
        self.history = []

    def _initialize_population(self):
        return [
            [random.uniform(*self.bounds) for _ in range(self.genome_length)]
            for _ in range(self.population_size)
        ]

    def _evaluate_individual(self, individual):
        # Ultra Intelligence: Combine multiple AI evaluators
        base_score = evaluate_strategy(individual)                        # Strategy performance
        dna_score = check_dna_uniqueness(individual)                      # Prevent duplicates
        fraud_penalty = 0.0 if not detect_fraud_pattern(individual) else 0.5  # Penalize suspicious patterns
        feedback_score = feedback_evaluator(individual)                  # Learn from loop feedback

        # Smart Weighted Score Fusion
        final_score = (
            (base_score * 0.4) +
            (dna_score * 0.2) +
            (feedback_score * 0.3) -
            (fraud_penalty * 0.5)
        )
        return max(0, final_score)

    def _evaluate_population(self, population):
        if self.parallel:
            with multiprocessing.Pool(self.threads) as pool:
                return pool.map(self._evaluate_individual, population)
        else:
            return [self._evaluate_individual(ind) for ind in population]

    def _select_parents(self, population, fitness_scores):
        total_fitness = sum(fitness_scores)
        probabilities = [f / total_fitness for f in fitness_scores]
        return random.choices(population, weights=probabilities, k=2)

    def _crossover(self, p1, p2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.genome_length - 1)
            return p1[:point] + p2[point:], p2[:point] + p1[point:]
        return p1, p2

    def _mutate(self, individual):
        new_individual = deepcopy(individual)
        for i in range(len(new_individual)):
            if random.random() < self.mutation_rate:
                new_individual[i] = random.uniform(*self.bounds)
        return new_individual

    def _evolve_population(self, population, fitness_scores):
        new_population = []

        if self.elitism:
            elite_idx = fitness_scores.index(max(fitness_scores))
            new_population.append(deepcopy(population[elite_idx]))

        while len(new_population) < self.population_size:
            parent1, parent2 = self._select_parents(population, fitness_scores)
            child1, child2 = self._crossover(parent1, parent2)
            new_population.append(self._mutate(child1))
            if len(new_population) < self.population_size:
                new_population.append(self._mutate(child2))

        return new_population

    def _log_generation(self, generation, best_score, best_individual):
        logger.info(f"🧬 Gen {generation} | Best Score: {best_score:.4f}")
        self.history.append({
            "generation": generation,
            "best_score": best_score,
            "best_individual": best_individual
        })

    def run(self):
        population = self._initialize_population()

        for gen in range(1, self.generations + 1):
            fitness = self._evaluate_population(population)
            best_idx = fitness.index(max(fitness))
            best_score = fitness[best_idx]
            best_individual = population[best_idx]

            self._log_generation(gen, best_score, best_individual)

            if gen % 10 == 0:
                self.mutation_rate *= 0.95  # Smart decay for convergence

            population = self._evolve_population(population, fitness)

        final_fitness = self._evaluate_population(population)
        final_best_idx = final_fitness.index(max(final_fitness))
        return population[final_best_idx], max(final_fitness)
