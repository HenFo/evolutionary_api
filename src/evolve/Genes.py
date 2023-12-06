
from abc import ABC, abstractclassmethod
from typing import Any, Callable

import numpy as np


class Gene(ABC):
    def __init__(self, allele:Any, mutation_rate:float) -> None:
        super().__init__()
        self.allele = allele
        self.mutation_rate = mutation_rate

    @abstractclassmethod
    def mutate(self, mutation_rate = None) -> "Gene":
        raise NotImplementedError

    def __str__(self) -> str:
        return str(self.allele)

    def __repr__(self) -> str:
        return f"Gene({str(self.allele)})"


class FloatGene(Gene):
    def __init__(self, allele:float = None, mutation_rate:float = 0.2, range:tuple = None, mutation_sd:float = 1) -> None:
        if range is None:
            allele = np.random.normal() if allele is None else allele
        else:
            allele = np.random.uniform(range[0], range[1]) if allele is None else allele

        super().__init__(allele, mutation_rate)
        self.range = (-np.inf, np.inf) if range is None else range
        self.mutation_sd = mutation_sd
    
    def mutate(self, mutation_rate = None) -> "FloatGene":
        mutation_rate = mutation_rate if mutation_rate is not None else self.mutation_rate
        if np.random.random() <= mutation_rate:
            mut_allele = self.allele + np.random.normal(scale=self.mutation_sd)
            mut_allele = mut_allele if mut_allele > self.range[0] else self.range[0]
            mut_allele = mut_allele if mut_allele < self.range[1] else self.range[1]
            return FloatGene(mut_allele, self.mutation_rate, self.range, self.mutation_sd)
        return self
    

class BinaryGene(Gene):
    def __init__(self, allele:int = None, mutation_rate:float = 0.2) -> None:
        allele = allele if allele is not None else 0 if np.random.random() < 0.5 else 1
        super().__init__(allele, mutation_rate)

    def mutate(self, mutation_rate = None) -> "BinaryGene":
        mutation_rate = mutation_rate if mutation_rate is not None else self.mutation_rate
        mut_allele = self.allele if np.random.random() > mutation_rate else (self.allele + 1) % 2
        return BinaryGene(mut_allele, self.mutation_rate)


class IntGene(Gene):
    def __init__(self, allele: int = None, mutation_rate:float = 0.2, range:tuple = None, mutation_span:float = (-5,5)) -> None:
        if range is None:
            allele = np.random.randint(0, 100) if allele is None else allele
        else:
            allele = np.random.randint(range[0], range[1]) if allele is None else allele

        super().__init__(allele, mutation_rate)
        self.range = range
        self.mutation_span = mutation_span
    
    def mutate(self, mutation_rate = None) -> "IntGene":
        mutation_rate = mutation_rate if mutation_rate is not None else self.mutation_rate
        mut_allele = self.allele if np.random.random() > mutation_rate \
            else self.allele + np.random.randint(self.mutation_span[0], self.mutation_span[1])
        mut_allele = mut_allele if mut_allele < self.range[1] else self.range[1]
        mut_allele = mut_allele if mut_allele > self.range[0] else self.range[0]

        return IntGene(mut_allele, self.mutation_rate, self.range, self.mutation_span)

class SymbolGene(Gene):
    def __init__(self, allele:str = None, mutation_rate:float = 0.2, symbols:list[str] = [], symbol_probs:list[float] = None) -> None:
        assert symbol_probs is None or len(symbols) == len(symbol_probs)
        
        allele = allele if allele is not None else np.random.choice(symbols)
        super().__init__(allele, mutation_rate)
        self.symbols = symbols
        self.symbol_probs = symbol_probs

    def mutate(self, mutation_rate = None) -> "SymbolGene":
        mutation_rate = mutation_rate if mutation_rate is not None else self.mutation_rate
        mut_allele = self.allele if np.random.random() > mutation_rate \
            else np.random.choice(self.symbols, p=self.symbol_probs)
        return SymbolGene(mut_allele, self.mutation_rate, self.symbols)
    
    def __str__(self) -> str:
        return self.allele

    def __repr__(self) -> str:
        return f"SymbolGene({str(self.allele)})"