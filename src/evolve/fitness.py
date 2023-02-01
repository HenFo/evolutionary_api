from typing import Callable
import numpy as np
from .Population import Chromosome

def fitness_function(func:Callable) -> Callable:
	def filter_calculated(chrom:Chromosome) -> float:
		if chrom.fitness_value < np.inf:
			return chrom.fitness_value
		return func(chrom)
	return filter_calculated