"""
@Project ：PyTorch-CycleGAN-master
@File    ：visual_3d.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/10/29 22:12
@Des     ：
"""
import matplotlib.pyplot as plt
import numpy as np



BODY_22_color = np.array(
    [[239, 43, 3], [3, 216, 239], [3, 216, 239], [3, 216, 239], [239, 219, 3], [239, 219, 3], [239, 219, 3]
        , [239, 3, 217], [239, 3, 217], [239, 3, 217], [37, 239, 3], [37, 239, 3], [37, 239, 3], [239, 43, 3]
        , [239, 43, 3], [239, 118, 3], [3, 239, 173], [3, 239, 173], [3, 239, 173], [125, 3, 239], [125, 3, 239]
        , [125, 3, 239]])

BODY_22_pairs = np.array(
    [[14, 13], [13, 0], [14, 19], [14, 16], [19, 20], [20, 21], [16, 17], [17, 18], [0, 1], [1, 2], [2, 3], [0, 7],
     [7, 8], [8, 9], [14, 15], [9, 11], [11, 12], [9, 10],
     [3, 5], [5, 6], [3, 4]])


def show_keypoints(keypoints_pred):
    fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.set_zlim(0, 200)
    ax.tick_params('x', labelbottom=True)
    ax.tick_params('y', labelleft=True)
    ax.tick_params('z', labelleft=True)
    ax.view_init(20, -70)
    xs = keypoints_pred[:, 0]
    ys = keypoints_pred[:, 1]
    zs = keypoints_pred[:, 2]
    for i in range(BODY_22_pairs.shape[0]):
        index_1 = BODY_22_pairs[i, 0]
        index_2 = BODY_22_pairs[i, 1]
        xs_line = [xs[index_1], xs[index_2]]
        ys_line = [ys[index_1], ys[index_2]]
        zs_line = [zs[index_1], zs[index_2]]
        ax.plot(xs_line, ys_line, zs_line, color=BODY_22_color[i] / 255.0)

    ax.scatter(xs, ys, zs, s=50, c=BODY_22_color[:22] / 255.0)
    # 显示图形
    plt.show()



