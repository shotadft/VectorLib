import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (for 3D projection)
from package.positionlib.position import Position
from package.vectorlib.vector import Vec3

def draw_vector3d_and_export(radius: int, pos: tuple[int, int, int], outdir: str = "") -> str:
    if not outdir:
        outdir = os.path.expanduser("~/Desktop")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(True)
    ax.quiver(0, 0, 0, pos[0], pos[1], pos[2], color='r', arrow_length_ratio=0.1)
    ax.scatter([0, pos[0]], [0, pos[1]], [0, pos[2]], color='k', marker='o')
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)
    ax.set_xticks(range(-radius, radius + 1))
    ax.set_yticks(range(-radius, radius + 1))
    ax.set_zticks(range(-radius, radius + 1))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    outpath = os.path.join(outdir, f"vector3d_r{radius}_({pos[0]},{pos[1]},{pos[2]}).png")
    fig.savefig(outpath)
    plt.close(fig)
    return outpath

if __name__ == "__main__":
    draw_vector3d_and_export(10, (3, 4, 5)) 