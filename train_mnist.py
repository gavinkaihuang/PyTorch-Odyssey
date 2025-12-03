import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time

# 1. 设置设备：优先使用 MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 当前使用的训练设备: {device}")

# 2. 准备数据 (第一次运行会自动下载)
# 定义数据预处理：转为 Tensor 并归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("⬇️ 正在下载/加载 MNIST 数据集...")
# 如果下载慢，可以多等一会儿，或者挂梯子
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# 定义数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 3. 定义一个简单的卷积神经网络 (CNN)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) # 卷积层 1
        self.conv2 = nn.Conv2d(32, 64, 3, 1) # 卷积层 2
        self.fc1 = nn.Linear(9216, 128)      # 全连接层 1
        self.fc2 = nn.Linear(128, 10)        # 输出层 (0-9 共10个数字)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x) # 激活函数
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2) # 池化
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# 4. 初始化模型并搬运到 MPS
model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=1.0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

# 5. 训练函数
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 关键步骤：把数据也搬运到 MPS 上
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] \tLoss: {loss.item():.6f}')

# 6. 开始训练
print("🔥 开始训练...")
start_time = time.time()

for epoch in range(1, 3): # 简单跑 2 轮试试
    train(epoch)
    scheduler.step()

end_time = time.time()
print(f"\n✅ 训练完成！总耗时: {end_time - start_time:.2f} 秒")