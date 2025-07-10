import os
import matplotlib.pyplot as plt
from package.positionlib.position import Position
from package.vectorlib.vector import Vec2

def draw_vector_and_export(radius: int, pos: tuple[int, int], outdir: str = "./") -> str:
    p = Position(*pos)
    v = Vec2(0, 0)
    arrow = Vec2(int(p.x), int(p.y)) - v
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect('equal')
    ax.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)
    ax.plot(0, 0, 'ko')
    ax.arrow(0, 0, arrow[0], arrow[1], head_width=radius*0.05, head_length=radius*0.08, fc='r', ec='r', length_includes_head=True)
    ax.set_xticks(range(-radius, radius+1, max(1, radius//8)))
    ax.set_yticks(range(-radius, radius+1, max(1, radius//8)))
    outpath = os.path.join(outdir, f'vector_draw_r{radius}_x{pos[0]}_y{pos[1]}.png')
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