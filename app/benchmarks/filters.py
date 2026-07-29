import random
from typing import List, Union, Optional
from app.benchmarks.models import UnifiedBenchmarkTask

class BenchmarkFilters:
    """
    Applies operations like benchmark-based filtering, shuffling, and sampling on lists of tasks.
    """
    
    @staticmethod
    def filter_by_benchmarks(
        tasks: List[UnifiedBenchmarkTask], 
        benchmarks: Union[str, List[str]]
    ) -> List[UnifiedBenchmarkTask]:
        """
        Filters tasks to run only specified benchmarks (single benchmark name or multiple).
        """
        if isinstance(benchmarks, str):
            target_benchmarks = {benchmarks.lower().strip()}
        else:
            target_benchmarks = {b.lower().strip() for b in benchmarks}
            
        return [t for t in tasks if t.benchmark.lower() in target_benchmarks]

    @staticmethod
    def shuffle(tasks: List[UnifiedBenchmarkTask], seed: Optional[int] = None) -> List[UnifiedBenchmarkTask]:
        """
        Shuffles tasks to randomize execution order. Supports random seed for evaluation reproducibility.
        """
        shuffled_tasks = list(tasks)
        if seed is not None:
            random.seed(seed)
        random.shuffle(shuffled_tasks)
        return shuffled_tasks

    @staticmethod
    def sample(tasks: List[UnifiedBenchmarkTask], n: int) -> List[UnifiedBenchmarkTask]:
        """
        Samples exactly N tasks from the provided task list.
        """
        if n <= 0:
            return []
        if n >= len(tasks):
            return list(tasks)
        return tasks[:n]
