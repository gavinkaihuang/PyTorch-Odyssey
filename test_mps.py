import torch
import time

# 1. 检查 MPS 是否可用
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 成功！MPS (Metal Performance Shaders) 加速已开启。")
    print(f"当前设备: {device}")
else:
    print("❌ 警告：未检测到 MPS，正在使用 CPU。请检查 macOS 版本或 PyTorch 版本。")
    device = torch.device("cpu")

# 2. 简单性能测试：矩阵乘法
# 创建两个较大的随机矩阵
N = 5000
x = torch.randn(N, N, device=device)
y = torch.randn(N, N, device=device)

print(f"正在 {device} 上进行 {N}x{N} 矩阵乘法计算...")
start_time = time.time()

# 这里的计算会在 GPU (MPS) 上执行
result = torch.matmul(x, y)

# 强制同步 (因为 GPU 计算是异步的，需要等待结果完成才能计算时间)
if device.type == "mps":
    torch.mps.synchronize()

end_time = time.time()
print(f"耗时: {end_time - start_time:.4f} 秒")