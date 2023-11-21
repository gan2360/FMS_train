"""
@Project ：PyTorch-CycleGAN-master
@File    ：visual_2d.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/10/29 22:13
@Des     ：
"""
import numpy as np
import matplotlib.pyplot as plt


BODY_22_color = np.array(
    [[239, 43, 3], [3, 216, 239], [3, 216, 239], [3, 216, 239], [239, 219, 3], [239, 219, 3], [239, 219, 3]
        , [239, 3, 217], [239, 3, 217], [239, 3, 217], [37, 239, 3], [37, 239, 3], [37, 239, 3], [239, 43, 3]
        , [239, 43, 3], [239, 118, 3], [3, 239, 173], [3, 239, 173], [3, 239, 173], [125, 3, 239], [125, 3, 239]
        , [125, 3, 239]])

BODY_22_pairs = np.array(
    [[14, 13], [13, 0], [14, 19], [14, 16], [19, 20], [20, 21], [16, 17], [17, 18], [0, 1], [1, 2], [2, 3], [0, 7],
     [7, 8], [8, 9], [14, 15], [9, 11], [11, 12], [9, 10],
     [3, 5], [5, 6], [3, 4]])


def show_2d_kpts(kpts):
    fig, ax = plt.subplots()
    ax.set_xlim(0,1280)
    ax.set_ylim(0, 720)
    ax.scatter(kpts[:, 0], kpts[:, 1], s=50, c=BODY_22_color[:22] / 255.0)
    for i in range(BODY_22_pairs.shape[0]):
        index1 = BODY_22_pairs[i, 0]
        index2 = BODY_22_pairs[i, 1]
        xs = [kpts[index1][0], kpts[index2][0]]
        ys = [kpts[index1][1], kpts[index2][1]]
        ax.plot(xs, ys, color=BODY_22_color[i] / 255.0)
    # ax.invert_yaxis()  # 翻转y轴以适应图像坐标系
    plt.show()



