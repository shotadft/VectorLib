import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (for 3D projection)
from package.positionlib.position import Position
from package.vectorlib.vector import Vec3

def draw_vector3d(radius: int, pos: tuple[int, int, int], outdir: str = "") -> str:
    if not outdir:
        outdir = os.path.expanduser("~/Desktop")
    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)

    ax.grid(True)
    ax.quiver(0, 0, 0, pos[0], pos[1], pos[2], color='r', arrow_length_ratio=0.1)
    ax.scatter(0, 0, 0, color='k', marker='o')
    ax.scatter(pos[0], pos[1], pos[2], color='k', marker='o')

    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)

    ax.set_xticks(range(-radius, radius + 1))
    ax.set_yticks(range(-radius, radius + 1))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

    outpath = os.path.join(outdir, f"vector3d_r{radius}_({pos[0]},{pos[1]},{pos[2]}).png")
    fig.savefig(outpath)
    plt.close(fig)

    return outpath

if __name__ == "__main__":
    try:
        MAX_RADIUS = 30
        radius = int(input(f"グリッド半径を0~{MAX_RADIUS}で指定してください: "))
        if not (0 <= radius <= MAX_RADIUS):
            raise ValueError
        x = int(input(f"x(整数: 0~{radius - 1}): "))
        y = int(input(f"y(整数: 0~{radius - 1}): "))
        z = int(input(f"z(整数: 0~{radius - 1}): "))

        print(f"({x}, {y}, {z})")
        out = draw_vector3d(radius, (x, y, z))
        print(f"Exported: {out}")
    except Exception as e:
        print(f"入力エラー: {e}")