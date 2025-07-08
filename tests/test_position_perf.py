import random
import time
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from package.positionlib.position import Position

def test_position_performance():
    random.seed(42)
    size = 256
    coords = [(random.randint(0, size-1), random.randint(0, size-1)) for _ in range(size * size)]

    positions = []
    t0 = time.perf_counter()
    for x, y in coords:
        positions.append(Position(x, y))
    t1 = time.perf_counter()
    print(f"[PERF] Position生成: {t1-t0:.4f}秒")

    t2 = time.perf_counter()
    zero_count = sum(p.is_zero() for p in positions)
    t3 = time.perf_counter()
    print(f"[PERF] is_zero判定: {t3-t2:.4f}秒 (0ベクトル数: {zero_count})")

    t4 = time.perf_counter()
    normalized = [p.normalize() for p in positions if not p.is_zero()]
    t5 = time.perf_counter()
    print(f"[PERF] normalize: {t5-t4:.4f}秒 (正規化数: {len(normalized)})")

    assert len(positions) == size * size
    assert all(isinstance(p, Position) for p in positions)

def test_massive_position_generation():
    random.seed(0xfa92023bc)
    N = 10_000
    size = 256
    sleep_interval = 0.5  # 秒
    next_sleep = time.perf_counter() + sleep_interval
    sleep_total = 0.0
    t0 = time.perf_counter()
    count = 0
    while count < N:
        x = random.randint(0, size-1)
        y = random.randint(0, size-1)
        _ = Position(x, y)
        count += 1
        now = time.perf_counter()
        if now >= next_sleep:
            sleep_start = time.perf_counter()
            time.sleep(sleep_interval)
            sleep_end = time.perf_counter()
            sleep_total += (sleep_end - sleep_start)
            next_sleep = time.perf_counter() + sleep_interval
    t1 = time.perf_counter()
    elapsed = t1 - t0 - sleep_total
    print(f"[PERF] ランダムPosition {N:,}回生成: {elapsed:.4f}秒（Sleep除外）")
    assert count == N 