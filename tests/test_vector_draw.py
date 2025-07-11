import os
import matplotlib.pyplot as plt
from package.positionlib.position import Position
from package.vectorlib.vector import Vec2

def draw_vector_and_export(radius: int, pos: tuple[int, int], outdir: str = "") -> str:
    if not outdir:
        outdir = os.path.expanduser("~/Desktop")
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.plot([0, pos[0]], [0, pos[1]], marker="o")
    ax.arrow(0, 0, pos[0], pos[1], head_width=0.2, head_length=0.3, fc="r", ec="r")
    ax.set_xticks(range(-radius, radius + 1))
    ax.set_yticks(range(-radius, radius + 1))
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    plt.show()
    outpath = os.path.join(outdir, f"vector_r{radius}_({pos[0]},{pos[1]}).png")
    fig.savefig(outpath)
    plt.close(fig)
    return outpath

if __name__ == "__main__":
    try:
        radius = int(input("グリッド半径を0〜255で指定してください: "))
        if not (0 <= radius <= 255):
            raise ValueError
        x = int(input("座標xを整数で指定してください: "))
        y = int(input("座標yを整数で指定してください: "))
        out = draw_vector_and_export(radius, (x, y))
        print(f"Exported: {out}")
    except Exception as e:
        print(f"入力エラー: {e}") 