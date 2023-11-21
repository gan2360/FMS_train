class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)




import torch
from torch.utils.data import DataLoader
import numpy as np

def calculate_mpjpe(model, test_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    mpjpe = 0.0
    num_samples = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 前向传播获取模型的预测结果
            outputs = model(inputs)

            # 计算MPJPE
            mpjpe += calculate_batch_mpjpe(outputs, targets)
            num_samples += 1

    # 计算平均MPJPE
    mpjpe /= num_samples

    return mpjpe

def calculate_batch_mpjpe(outputs, targets):
    # 在这里实现批量计算MPJPE的逻辑
    # outputs 和 targets 的形状都是 (batch_size, num_joints, 3)
    # 可以根据需要自定义计算MPJPE的方法

    # 这里只是一个示例，计算每个关节的欧氏距离并取平均作为MPJPE
    errors = torch.norm(outputs - targets, dim=2)
    mpjpe = errors.mean()

    return mpjpe.item()

# 示例用法
# model = YourModel()  # 替换为你的模型实例
# test_dataset = YourTestDataset()  # 替换为你的测试集实例
#
# mpjpe = calculate_mpjpe(model, test_dataset)
# print(f"Mean Per Joint Position Error (MPJPE): {mpjpe}"))