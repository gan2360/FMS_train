"""
@Project ：PyTorch-CycleGAN-master
@File    ：visual_pressure.py
@IDE     ：PyCharm
@Author  ：FMS
@Date    ：2023/10/29 22:12
@Des     ：
"""
import numpy as np
import matplotlib.pyplot as plt
def show_pressure(pressure):

    # 创建压力矩阵
    pressure_matrix = pressure

    # 绘制热图
    plt.imshow(pressure_matrix, cmap='hot', origin='lower')

    # 添加颜色条
    plt.colorbar()

    # 添加标题和轴标签
    plt.title('Pressure Distribution')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 显示图形
    plt.show()

