# 標準ライブラリ
import os

# サードパーティライブラリ
import matplotlib.pyplot as plt

# ローカルモジュール
from package.vecposlib.positionlib import Position
from package.vecposlib.vectorlib import Vec2

def draw_vector(radius: int, pos: Position[int], outdir: str = "") -> str:
    if not outdir:
        outdir = os.path.expanduser("~/Desktop")
    
    vec = Vec2(pos.x, pos.y)
    
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.plot([0, vec.x], [0, vec.y], marker="o")
    ax.arrow(0, 0, vec.x, vec.y, head_width=0.2, head_length=0.3, fc="r", ec="r")
    ax.set_xticks(range(-radius, radius + 1))
    ax.set_yticks(range(-radius, radius + 1))
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    plt.show()
    outpath = os.path.join(outdir, f"vector_r{radius}_({vec.x},{vec.y}).png")
    fig.savefig(outpath)
    plt.close(fig)
    return outpath

if __name__ == "__main__":
    try:
        MAX_RADIUS = 256
        radius = int(input(f"グリッド半径を0~{MAX_RADIUS}で指定してください: "))
        if not (0 <= radius <= MAX_RADIUS):
            raise ValueError
        x = int(input(f"x(整数: 0~{radius - 1}): "))
        y = int(input(f"y(整数: 0~{radius - 1}): "))

        pos = Position(x, y)
        print(f"Position: {pos}")
        out = draw_vector(radius, pos)
        print(f"Exported: {out}")
    except Exception as e:
        print(f"入力エラー: {e}")
