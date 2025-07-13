# 標準ライブラリ
import random
import statistics
import time
from dataclasses import dataclass
from typing import List

# ローカルモジュール
from package.vecposlib.positionlib import Position

@dataclass
class BenchmarkResult:
    """ベンチマーク結果を格納するデータクラス"""
    operation: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    throughput: float

class PositionBenchmark:
    """Positionクラスのベンチマーク実行クラス"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.results: List[BenchmarkResult] = []
    
    def run_benchmark(self, operation_name: str, iterations: int, 
                     operation_func, *args, **kwargs) -> BenchmarkResult:
        """単一のベンチマークを実行"""
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            operation_func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        throughput = iterations / total_time if total_time > 0 else 0.0
        
        result = BenchmarkResult(
            operation=operation_name,
            iterations=iterations,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_dev=std_dev,
            throughput=throughput
        )
        
        self.results.append(result)
        return result
    
    def print_results(self):
        """ベンチマーク結果を表示"""
        print("\n" + "="*80)
        print("POSITION CLASS BENCHMARK RESULTS")
        print("="*80)
        
        for result in self.results:
            print(f"\n{result.operation}")
            print("-" * len(result.operation))
            print(f"  Iterations: {result.iterations:,}")
            print(f"  Total Time: {result.total_time:.6f}s")
            print(f"  Average Time: {result.avg_time:.9f}s")
            print(f"  Min Time: {result.min_time:.9f}s")
            print(f"  Max Time: {result.max_time:.9f}s")
            print(f"  Std Dev: {result.std_dev:.9f}s")
            print(f"  Throughput: {result.throughput:,.0f} ops/sec")
        
        print("\n" + "="*80)

def test_position_creation_benchmark():
    """Position作成のベンチマーク"""
    benchmark = PositionBenchmark(seed=42)
    
    benchmark.run_benchmark(
        "1D Position Creation",
        100_000,
        lambda: Position(random.randint(0, 1000))
    )
    
    benchmark.run_benchmark(
        "2D Position Creation",
        100_000,
        lambda: Position(random.randint(0, 1000), random.randint(0, 1000))
    )
    
    benchmark.run_benchmark(
        "3D Position Creation",
        100_000,
        lambda: Position(random.randint(0, 1000), random.randint(0, 1000), random.randint(0, 1000))
    )
    
    benchmark.run_benchmark(
        "4D Position Creation",
        100_000,
        lambda: Position(random.randint(0, 1000), random.randint(0, 1000), 
                       random.randint(0, 1000), random.randint(0, 1000))
    )
    
    benchmark.print_results()

def test_position_property_benchmark():
    """Positionプロパティアクセスのベンチマーク"""
    benchmark = PositionBenchmark(seed=42)
    
    pos_2d = Position(100, 200)
    pos_3d = Position(100, 200, 300)
    pos_4d = Position(100, 200, 300, 400)
    
    benchmark.run_benchmark(
        "X Property Access",
        1_000_000,
        lambda: pos_2d.x
    )
    
    benchmark.run_benchmark(
        "Y Property Access",
        1_000_000,
        lambda: pos_2d.y
    )
    
    benchmark.run_benchmark(
        "Z Property Access (3D)",
        1_000_000,
        lambda: pos_3d.z
    )
    
    benchmark.run_benchmark(
        "W Property Access (4D)",
        1_000_000,
        lambda: pos_4d.w
    )
    
    benchmark.run_benchmark(
        "NDim Property Access",
        1_000_000,
        lambda: pos_2d.ndim
    )
    
    benchmark.print_results()

def test_position_method_benchmark():
    """Positionメソッドのベンチマーク"""
    benchmark = PositionBenchmark(seed=42)
    
    positions = [
        Position(0, 0),
        Position(3, 4),
        Position(1, 1, 1),
        Position(1, 1, 1, 1),
    ]
    
    for i, pos in enumerate(positions):
        benchmark.run_benchmark(
            f"is_zero() - {pos.ndim}D",
            100_000,
            lambda p=pos: p.is_zero()
        )
    
    for i, pos in enumerate(positions):
        benchmark.run_benchmark(
            f"to_list() - {pos.ndim}D",
            100_000,
            lambda p=pos: p.to_list()
        )
    
    for i, pos in enumerate(positions):
        benchmark.run_benchmark(
            f"to_tuple() - {pos.ndim}D",
            100_000,
            lambda p=pos: p.to_tuple()
        )
    
    non_zero_positions = [pos for pos in positions if not pos.is_zero()]
    for i, pos in enumerate(non_zero_positions):
        benchmark.run_benchmark(
            f"normalize() - {pos.ndim}D",
            10_000,
            lambda p=pos: p.normalize()
        )
    
    benchmark.print_results()

def test_position_bulk_operations_benchmark():
    """Positionの一括操作のベンチマーク"""
    benchmark = PositionBenchmark(seed=42)
    
    size = 1000
    coords_2d = [(random.randint(0, 255), random.randint(0, 255)) for _ in range(size)]
    coords_3d = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(size)]
    
    benchmark.run_benchmark(
        "Bulk 2D Position Creation",
        size,
        lambda: [Position(x, y) for x, y in coords_2d]
    )
    
    benchmark.run_benchmark(
        "Bulk 3D Position Creation",
        size,
        lambda: [Position(x, y, z) for x, y, z in coords_3d]
    )
    
    positions_2d = [Position(x, y) for x, y in coords_2d]
    positions_3d = [Position(x, y, z) for x, y, z in coords_3d]
    
    benchmark.run_benchmark(
        "Bulk is_zero() - 2D",
        size,
        lambda: [p.is_zero() for p in positions_2d]
    )
    
    benchmark.run_benchmark(
        "Bulk is_zero() - 3D",
        size,
        lambda: [p.is_zero() for p in positions_3d]
    )
    
    non_zero_2d = [p for p in positions_2d if not p.is_zero()]
    non_zero_3d = [p for p in positions_3d if not p.is_zero()]
    
    if non_zero_2d:
        benchmark.run_benchmark(
            "Bulk normalize() - 2D",
            len(non_zero_2d),
            lambda: [p.normalize() for p in non_zero_2d]
        )
    
    if non_zero_3d:
        benchmark.run_benchmark(
            "Bulk normalize() - 3D",
            len(non_zero_3d),
            lambda: [p.normalize() for p in non_zero_3d]
        )
    
    benchmark.print_results()

def test_position_memory_benchmark():
    """Positionのメモリ使用量のベンチマーク"""
    import gc
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    def get_memory_usage():
        """現在のメモリ使用量を取得"""
        return process.memory_info().rss / 1024 / 1024  # MB
    
    benchmark = PositionBenchmark(seed=42)
    
    gc.collect()
    initial_memory = get_memory_usage()
    
    positions = []
    for _ in range(100_000):
        positions.append(Position(random.randint(0, 255), random.randint(0, 255)))
    
    gc.collect()
    final_memory = get_memory_usage()
    memory_used = final_memory - initial_memory
    
    print(f"\nMemory Benchmark Results:")
    print(f"Initial Memory: {initial_memory:.2f} MB")
    print(f"Final Memory: {final_memory:.2f} MB")
    print(f"Memory Used: {memory_used:.2f} MB")
    print(f"Memory per Position: {memory_used * 1024 * 1024 / 100_000:.2f} bytes")
    
    del positions
    gc.collect()

def test_position_comparison_benchmark():
    """Positionの比較操作のベンチマーク"""
    benchmark = PositionBenchmark(seed=42)
    
    positions = [
        Position(0, 0),
        Position(1, 1),
        Position(2, 2),
        Position(3, 4),
        Position(5, 12),
    ]
    
    benchmark.run_benchmark(
        "Equality Comparison",
        100_000,
        lambda: positions[0] == positions[1]
    )
    
    benchmark.run_benchmark(
        "String Representation",
        100_000,
        lambda: str(positions[0])
    )
    
    benchmark.run_benchmark(
        "List Conversion",
        100_000,
        lambda: positions[0].to_list()
    )
    
    benchmark.print_results()

if __name__ == "__main__":
    print("Running Position Benchmark Tests...")
    
    test_position_creation_benchmark()
    test_position_property_benchmark()
    test_position_method_benchmark()
    test_position_bulk_operations_benchmark()
    test_position_comparison_benchmark()
    
    try:
        test_position_memory_benchmark()
    except ImportError:
        print("psutil not available, skipping memory benchmark")
    
    print("\nAll benchmark tests completed!")
