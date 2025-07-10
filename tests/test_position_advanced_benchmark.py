import random
import time
import pytest
import sys
import os
import threading
import concurrent.futures
import statistics
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from package.positionlib.position import Position

@dataclass
class StressTestResult:
    """ストレステスト結果を格納するデータクラス"""
    test_name: str
    iterations: int
    total_time: float
    avg_time: float
    memory_usage: float
    success_rate: float
    errors: List[str]

class AdvancedPositionBenchmark:
    """高度なPositionクラスのベンチマーク実行クラス"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.stress_results: List[StressTestResult] = []
    
    def run_stress_test(self, test_name: str, iterations: int, 
                       test_func, *args, **kwargs) -> StressTestResult:
        """ストレステストを実行"""
        times = []
        errors = []
        start_time = time.perf_counter()
        
        for i in range(iterations):
            try:
                iter_start = time.perf_counter()
                test_func(*args, **kwargs)
                iter_end = time.perf_counter()
                times.append(iter_end - iter_start)
            except Exception as e:
                errors.append(f"Iteration {i}: {str(e)}")
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        success_rate = (iterations - len(errors)) / iterations * 100
        avg_time = statistics.mean(times) if times else 0.0
        
        # メモリ使用量の概算（簡易版）
        memory_usage = len(times) * 8  # 概算値
        
        result = StressTestResult(
            test_name=test_name,
            iterations=iterations,
            total_time=total_time,
            avg_time=avg_time,
            memory_usage=memory_usage,
            success_rate=success_rate,
            errors=errors
        )
        
        self.stress_results.append(result)
        return result
    
    def print_stress_results(self):
        """ストレステスト結果を表示"""
        print("\n" + "="*80)
        print("ADVANCED POSITION STRESS TEST RESULTS")
        print("="*80)
        
        for result in self.stress_results:
            print(f"\n{result.test_name}")
            print("-" * len(result.test_name))
            print(f"  Iterations: {result.iterations:,}")
            print(f"  Total Time: {result.total_time:.6f}s")
            print(f"  Average Time: {result.avg_time:.9f}s")
            print(f"  Success Rate: {result.success_rate:.2f}%")
            print(f"  Errors: {len(result.errors)}")
            if result.errors:
                print(f"  Error Examples: {result.errors[:3]}")
        
        print("\n" + "="*80)

def test_position_stress_creation():
    """Position作成のストレステスト"""
    benchmark = AdvancedPositionBenchmark(seed=42)
    
    def create_random_position():
        """ランダムなPositionを作成"""
        dims = random.randint(1, 4)
        coords = [random.randint(-1000, 1000) for _ in range(dims)]
        return Position(*coords)
    
    # 大量作成ストレステスト
    benchmark.run_stress_test(
        "Massive Position Creation",
        1_000_000,
        create_random_position
    )
    
    # 浮動小数点Position作成ストレステスト
    def create_float_position():
        """浮動小数点のPositionを作成"""
        dims = random.randint(1, 4)
        coords = [random.uniform(-1000, 1000) for _ in range(dims)]
        return Position(*coords)
    
    benchmark.run_stress_test(
        "Float Position Creation",
        500_000,
        create_float_position
    )
    
    benchmark.print_stress_results()

def test_position_concurrent_operations():
    """Positionの並行処理テスト"""
    benchmark = AdvancedPositionBenchmark(seed=42)
    
    def concurrent_position_creation(thread_id: int, count: int):
        """並行してPositionを作成"""
        positions = []
        for i in range(count):
            x = random.randint(0, 255) + thread_id
            y = random.randint(0, 255) + thread_id
            positions.append(Position(x, y))
        return len(positions)
    
    def run_concurrent_test(thread_count: int, positions_per_thread: int):
        """並行テストを実行"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [
                executor.submit(concurrent_position_creation, i, positions_per_thread)
                for i in range(thread_count)
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        return sum(results)
    
    # 2スレッド並行テスト
    benchmark.run_stress_test(
        "Concurrent Creation - 2 Threads",
        100,
        lambda: run_concurrent_test(2, 1000)
    )
    
    # 4スレッド並行テスト
    benchmark.run_stress_test(
        "Concurrent Creation - 4 Threads",
        100,
        lambda: run_concurrent_test(4, 1000)
    )
    
    # 8スレッド並行テスト
    benchmark.run_stress_test(
        "Concurrent Creation - 8 Threads",
        50,
        lambda: run_concurrent_test(8, 1000)
    )
    
    benchmark.print_stress_results()

def test_position_edge_cases():
    """Positionのエッジケーステスト"""
    benchmark = AdvancedPositionBenchmark(seed=42)
    
    def test_zero_vector():
        """ゼロベクトルのテスト"""
        pos = Position(0, 0)
        assert pos.is_zero()
        try:
            pos.normalize()
            assert False, "Should raise ValueError for zero vector"
        except ValueError:
            pass
    
    benchmark.run_stress_test(
        "Zero Vector Operations",
        10_000,
        test_zero_vector
    )
    
    def test_large_numbers():
        """大きな数値のテスト"""
        large_pos = Position(1e10, 1e10, 1e10)
        assert not large_pos.is_zero()
        normalized = large_pos.normalize()
        assert abs(normalized.x - 0.5773502691896258) < 1e-10
    
    benchmark.run_stress_test(
        "Large Number Operations",
        1_000,
        test_large_numbers
    )
    
    def test_small_numbers():
        """小さな数値のテスト"""
        small_pos = Position(1e-10, 1e-10, 1e-10)
        assert not small_pos.is_zero()
        normalized = small_pos.normalize()
        assert abs(normalized.x - 0.5773502691896258) < 1e-10
    
    benchmark.run_stress_test(
        "Small Number Operations",
        1_000,
        test_small_numbers
    )
    
    benchmark.print_stress_results()

def test_position_memory_stress():
    """Positionのメモリストレステスト"""
    benchmark = AdvancedPositionBenchmark(seed=42)
    
    def create_and_destroy_positions():
        """Positionを作成して即座に破棄"""
        for _ in range(1000):
            pos = Position(random.randint(0, 255), random.randint(0, 255))
            _ = pos.x, pos.y, pos.is_zero()
            del pos
    
    benchmark.run_stress_test(
        "Memory Stress - Create and Destroy",
        100,
        create_and_destroy_positions
    )
    
    def create_large_list():
        """大きなリストを作成"""
        positions = []
        for _ in range(10_000):
            positions.append(Position(random.randint(0, 255), random.randint(0, 255)))
        # リストを保持したまま終了（メモリリークテスト）
        return len(positions)
    
    benchmark.run_stress_test(
        "Memory Stress - Large List",
        10,
        create_large_list
    )
    
    benchmark.print_stress_results()

def test_position_data_type_performance():
    """異なるデータ型でのパフォーマンステスト"""
    benchmark = AdvancedPositionBenchmark(seed=42)
    
    def test_int_positions():
        """整数Positionのテスト"""
        for _ in range(1000):
            pos = Position(random.randint(0, 255), random.randint(0, 255))
            _ = pos.x, pos.y, pos.is_zero()
    
    benchmark.run_stress_test(
        "Integer Position Performance",
        100,
        test_int_positions
    )
    
    def test_float_positions():
        """浮動小数点Positionのテスト"""
        for _ in range(1000):
            pos = Position(random.uniform(0, 255), random.uniform(0, 255))
            _ = pos.x, pos.y, pos.is_zero()
    
    benchmark.run_stress_test(
        "Float Position Performance",
        100,
        test_float_positions
    )
    
    def test_mixed_positions():
        """混合データ型Positionのテスト"""
        for _ in range(1000):
            if random.random() < 0.5:
                pos = Position(random.randint(0, 255), random.randint(0, 255))
            else:
                pos = Position(random.uniform(0, 255), random.uniform(0, 255))
            _ = pos.x, pos.y, pos.is_zero()
    
    benchmark.run_stress_test(
        "Mixed Data Type Performance",
        100,
        test_mixed_positions
    )
    
    benchmark.print_stress_results()

def test_position_dimension_performance():
    """異なる次元でのパフォーマンステスト"""
    benchmark = AdvancedPositionBenchmark(seed=42)
    
    def test_1d_positions():
        """1次元Positionのテスト"""
        for _ in range(1000):
            pos = Position(random.randint(0, 255))
            _ = pos.x, pos.ndim, pos.is_zero()
    
    benchmark.run_stress_test(
        "1D Position Performance",
        100,
        test_1d_positions
    )
    
    def test_2d_positions():
        """2次元Positionのテスト"""
        for _ in range(1000):
            pos = Position(random.randint(0, 255), random.randint(0, 255))
            _ = pos.x, pos.y, pos.ndim, pos.is_zero()
    
    benchmark.run_stress_test(
        "2D Position Performance",
        100,
        test_2d_positions
    )
    
    def test_3d_positions():
        """3次元Positionのテスト"""
        for _ in range(1000):
            pos = Position(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            _ = pos.x, pos.y, pos.z, pos.ndim, pos.is_zero()
    
    benchmark.run_stress_test(
        "3D Position Performance",
        100,
        test_3d_positions
    )
    
    def test_4d_positions():
        """4次元Positionのテスト"""
        for _ in range(1000):
            pos = Position(random.randint(0, 255), random.randint(0, 255), 
                         random.randint(0, 255), random.randint(0, 255))
            _ = pos.x, pos.y, pos.z, pos.w, pos.ndim, pos.is_zero()
    
    benchmark.run_stress_test(
        "4D Position Performance",
        100,
        test_4d_positions
    )
    
    benchmark.print_stress_results()

def test_position_operation_chains():
    """Position操作の連鎖テスト"""
    benchmark = AdvancedPositionBenchmark(seed=42)
    
    def test_operation_chain():
        """操作の連鎖をテスト"""
        pos = Position(random.randint(1, 255), random.randint(1, 255))
        # 連鎖的な操作
        coords = pos.to_list()
        pos2 = Position(*coords)
        normalized = pos2.normalize()
        coords2 = normalized.to_tuple()
        pos3 = Position(*coords2)
        return pos3.ndim
    
    benchmark.run_stress_test(
        "Operation Chain Performance",
        10_000,
        test_operation_chain
    )
    
    def test_complex_operations():
        """複雑な操作をテスト"""
        positions = []
        for _ in range(100):
            pos = Position(random.randint(1, 255), random.randint(1, 255))
            if not pos.is_zero():
                normalized = pos.normalize()
                positions.append(normalized)
        
        # 一括操作
        coords_list = [p.to_list() for p in positions]
        tuples_list = [p.to_tuple() for p in positions]
        return len(coords_list) + len(tuples_list)
    
    benchmark.run_stress_test(
        "Complex Operations Performance",
        100,
        test_complex_operations
    )
    
    benchmark.print_stress_results()

if __name__ == "__main__":
    # 高度なベンチマークテストの実行
    print("Running Advanced Position Benchmark Tests...")
    
    test_position_stress_creation()
    test_position_concurrent_operations()
    test_position_edge_cases()
    test_position_memory_stress()
    test_position_data_type_performance()
    test_position_dimension_performance()
    test_position_operation_chains()
    
    print("\nAll advanced benchmark tests completed!") 