import os
from functools import total_ordering
from typing import Callable

import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from .Genes import Gene


@total_ordering
class Chromosome(list):
    def __init__(self, genes:list[Gene]) -> None:
        assert all(map(lambda x: issubclass(type(x), Gene), genes)), "Invalid types. Should all be of type 'Gene'"
        super().__init__(genes)
        self.fitness_value:float = np.inf

    def mutate(self, mutation_rate = None) -> "Chromosome":
        return Chromosome([x.mutate(mutation_rate) for x in self])

    def as_tuple(self) -> tuple:
        return tuple(x.allele for x in self)

    def set_fitness(self, fitness_value:float) -> None:
        self.fitness_value = fitness_value

    @classmethod
    def create_offsprings(self, p1:"Chromosome", p2:"Chromosome", split_points:list[int] = None) -> tuple["Chromosome", "Chromosome"]:
        assert len(p1) == len(p2), "Chromosomes must be same length"
        split_points = split_points if split_points is not None else [int(len(p1)/2)]
        assert max(split_points) < len(p1), f"Split point {max(split_points)} too large for Chromosome of length {len(p1)}"

        def _split_chromosome(l:list, s:list) -> list[list]:
            start = 0
            splits = []
            for end in s:
                splits.append(l[start:end])
                start = end
            splits.append(l[start:])
            return splits

        def _join_chromosomes(l1:list[list], l2:list[list]) -> tuple["Chromosome", "Chromosome"]:
            a, b = [], []
            for i in range(0, len(l1), 2):
                a += l1[i]
                b += l2[i]
                if i+1 < len(l1):
                    a += l2[i+1]
                    b += l1[i+1]
            return Chromosome(a), Chromosome(b)
        
        # Split chromosomes along the split points
        c_split1, c_split2 = _split_chromosome(p1, split_points), _split_chromosome(p2, split_points)
        off1, off2 = _join_chromosomes(c_split1, c_split2)

        return (off1, off2)
    
    def __repr__(self) -> str:
        return f"Chromo({self.fitness_value}, {super().__repr__()})"

    def __lt__(self, other:"Chromosome") -> bool:
        return self.fitness_value < other.fitness_value
    
    def __eq__(self, other:"Chromosome") -> bool:
        return self.fitness_value == other.fitness_value
    
    def __hash__(self):
        return hash(self.as_tuple())

    def __add__(self, other:"Chromosome") -> float:
        return self.fitness_value + other.fitness_value


class Population(list):
    def __init__(self, chromosomes: list[Chromosome], fitness_func:Callable = None) -> None:
        assert all(map(lambda x: type(x) == Chromosome, chromosomes)), "Invalid types. Should all be of type 'Gene'"
        assert all([len(lst) == len(chromosomes[0]) for lst in chromosomes]), "Chromosomes don't have the same length"
        super().__init__(chromosomes)

        self.fitness_func = fitness_func
        self.sorted = False
        self.rank_probabilities = self._calc_rank_probability()

    def sort(self) -> "Population":
        if not self.sorted:
            super().sort(key=lambda x: x.fitness_value)
            self.sorted = True
        return self
    
    def _calc_rank_probability(self) -> np.ndarray:
        a = np.arange(len(self))[::-1]
        return np.array([(i+1)/(len(self)*(len(self)+1)/2) for i in a])

    def evolve(self, strategy:str = "2,2", max_iter:int = 100, precision:float = 1e-2, patience:int = 10, parallel:bool = False, fill_duplicates:bool = True) -> tuple[dict,"Population"]:
        assert strategy in ("2+2", "2,2")
        if strategy == "2+2":
            return self._evolve_parent_and_child(max_iter, precision, patience, parallel, fill_duplicates)
        if strategy == "2,2":
            return self._evolve_children_only(max_iter, precision, parallel)
    
    @staticmethod
    def _evolve_population(sorted_pop:"Population") -> "Population":
            offsprings:list[Chromosome] = list()
            # print("current best error =", sorted_pop[0].fitness_value)
            while len(offsprings) < len(sorted_pop):
                p1, p2 = np.random.choice(len(sorted_pop), size=2, p=sorted_pop.rank_probabilities)
                p1 = sorted_pop[p1]
                p2 = sorted_pop[p2]
                off1, off2 = Chromosome.create_offsprings(p1,p2)
                offsprings.append(off1.mutate())
                offsprings.append(off2.mutate())

            return Population(offsprings, sorted_pop.fitness_func)


    def _evolve_children_only(self, max_iter:int = 100, precision:float = 1e-2, parallel:bool = False) -> "Population":
        current_pop = self
        history = {}
        for _ in tqdm(range(max_iter), desc="Evolving", position=0, leave=True):
            current_pop.calc_fitness(parallel)
            if i % 50 == 0:
                tqdm.write(f"Current best error = {current_pop[0].fitness_value}")
            if current_pop[0].fitness_value <= precision:
                break
            for i, chrom in enumerate(current_pop):
                history[i] = history.get(i, []) + [chrom.fitness_value]
            current_pop = Population._evolve_population(current_pop)
        
        return history, current_pop


    def _evolve_parent_and_child(self, max_iter:int = 100, precision:float = 1e-2, patience:int = 10, parallel:bool = False, fill_duplicates:bool = True) -> "Population":
        current_pop = self
        history = {}
        current_pop.calc_fitness(parallel)
        for i in tqdm(range(max_iter), desc="Evolving", position=0, leave=True):
            if i % 50 == 0:
                tqdm.write(f"Current best error = {current_pop[0].fitness_value}")
            if current_pop[0].fitness_value <= precision:
                break
            for i, chrom in enumerate(current_pop):
                history[i] = history.get(i, []) + [chrom.fitness_value]
            evolved_pop = Population._evolve_population(current_pop)
            evolved_pop.calc_fitness(parallel)
            current_pop = current_pop.merge(evolved_pop, fill_duplicates=fill_duplicates)
            current_pop.sort()
        
        return history, current_pop

    def calc_fitness(self, parallel:bool = False) -> None:
        assert self.fitness_func is not None
        if not parallel:
            for x in tqdm(self, position=1, desc="Population", leave=False):
                fitness = self.fitness_func(x)
                x.set_fitness(fitness)
        else:
            fitness_vals = process_map(self.fitness_func, self, max_workers=min(32, os.cpu_count() + 4, len(self)))
            for x, f in zip(self, fitness_vals):
                x.set_fitness(f)
        self.sort()
    
    @staticmethod
    def remove_duplicates(population: "Population") -> "Population":
        return Population(list(set(population)), population.fitness_func)

    
    def __repr__(self) -> str:
        return "Population(\n\t" + "\n\t".join([repr(x) for x in self]) + "\n)"

    
    def merge(self, other:"Population", ratio:float = 0.5, fill_duplicates:bool = True) -> "Population":
        split = int(len(self) * ratio)
        new_pop = self[:split] + other[:-split]
        if fill_duplicates:
            new_pop = Population.remove_duplicates(new_pop)
            while len(new_pop) < len(self):
                new_pop.append(self[0].mutate(1))
        assert len(new_pop) == len(self)
        return Population(new_pop, self.fitness_func)

    def __add__(self, other:"Population") -> "Population":
        return Population(list(self) + list(other), self.fitness_func)
    
    def __getitem__(self, index):
        # Check if slicing is being performed
        if isinstance(index, slice):
            # Create an instance of your subclass and populate it with the sliced elements
            pop = Population(super().__getitem__(index), self.fitness_func)
            pop.sorted = self.sorted
            return pop
        else:
            # Return the individual element
            return super().__getitem__(index)