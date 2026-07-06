# !pip install paddlepaddle-gpu
import os
import time
import paddle
import paddle.nn as nn
from paddle.vision.datasets import Cifar10
import paddle.vision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

# 【Kaggle 优化1】明确指定 Kaggle 的可写输出目录
OUTPUT_DIR = "/kaggle/working/gan_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. 定义生成器 (Generator) 
# ==========================================
class Generator(nn.Layer):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2DTranspose(100, 256, 4, 1, 0, bias_attr=False),
            nn.BatchNorm2D(256),
            nn.ReLU(True),
            nn.Conv2DTranspose(256, 128, 4, 2, 1, bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(True),
            nn.Conv2DTranspose(128, 64, 4, 2, 1, bias_attr=False),
            nn.BatchNorm2D(64),
            nn.ReLU(True),
            nn.Conv2DTranspose(64, 3, 4, 2, 1, bias_attr=False),
            nn.Tanh()
        )
    def forward(self, noise):
        return self.main(noise)

# ==========================================
# 2. 定义判别器 (Discriminator)
# ==========================================
class Discriminator(nn.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2D(3, 64, 4, 2, 1, bias_attr=False),
            nn.LeakyReLU(0.2),
            nn.Conv2D(64, 128, 4, 2, 1, bias_attr=False),
            nn.BatchNorm2D(128),
            nn.LeakyReLU(0.2),
            nn.Conv2D(128, 256, 4, 2, 1, bias_attr=False),
            nn.BatchNorm2D(256),
            nn.LeakyReLU(0.2),
            nn.Conv2D(256, 1, 4, 1, 0, bias_attr=False),
            nn.Sigmoid()
        )
    def forward(self, img):
        return self.main(img)

# ==========================================
# 辅助函数：保存网格图片
# ==========================================
def save_generated_images(images, epoch, path=OUTPUT_DIR):
    images = images.numpy()
    images = (images * 0.5) + 0.5
    images = np.clip(images, 0, 1)
    
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    idx = 0
    for i in range(4):
        for j in range(4):
            img = np.transpose(images[idx], (1, 2, 0))
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            idx += 1
    plt.tight_layout()
    plt.savefig(f"{path}/epoch_{epoch}_generated.png")
    plt.close()
    print(f"成功保存第 {epoch} 阶段图片到: {path}/epoch_{epoch}_generated.png")

# ==========================================
# 3. 数据集加载与预处理
# ==========================================
print("正在下载并加载 CIFAR-10 数据集...")
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
train_dataset = Cifar10(mode='train', transform=transform, download=True)
dataloader = paddle.io.DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
print("数据集加载成功！")

# ==========================================
# 4. 实例化模型、损失函数与优化器
# ==========================================
netG = Generator()
netD = Discriminator()
criterion = nn.BCELoss()

lr = 0.0002
optimizerD = paddle.optimizer.Adam(learning_rate=lr, parameters=netD.parameters(), beta1=0.5, beta2=0.999)
optimizerG = paddle.optimizer.Adam(learning_rate=lr, parameters=netG.parameters(), beta1=0.5, beta2=0.999)

fixed_noise = paddle.randn([16, 100, 1, 1], dtype='float32')

# 【Kaggle 优化2】新增列表用于记录每一轮的平均 Loss，方便后面画图
d_losses = []
g_losses = []

# ==========================================
# 5. 开启训练循环
# ==========================================
NUM_EPOCHS = 200  
print(f"开始进入对抗训练循环，总共 {NUM_EPOCHS} 轮。")

for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    epoch_d_loss = 0.0
    epoch_g_loss = 0.0
    batch_count = 0
    
    for batch_id, data in enumerate(dataloader):
        real_imgs, _ = data
        current_batch_size = real_imgs.shape[0]
        
        label_real = paddle.ones([current_batch_size, 1, 1, 1], dtype='float32')
        label_fake = paddle.zeros([current_batch_size, 1, 1, 1], dtype='float32')
        
        # 训练判别器 D
        output_real = netD(real_imgs)
        loss_D_real = criterion(output_real, label_real)
        
        noise = paddle.randn([current_batch_size, 100, 1, 1], dtype='float32')
        fake_imgs = netG(noise)
        output_fake = netD(fake_imgs.detach())
        loss_D_fake = criterion(output_fake, label_fake)
        
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        optimizerD.step()
        optimizerD.clear_grad()
        
        # 训练生成器 G
        output_fake_for_G = netD(fake_imgs)
        loss_G = criterion(output_fake_for_G, label_real)
        loss_G.backward()
        optimizerG.step()
        optimizerG.clear_grad()
        
        # 累加 Loss
        epoch_d_loss += float(loss_D)
        epoch_g_loss += float(loss_G)
        batch_count += 1
        
        if batch_id % 100 == 0:
            print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}] | Batch [{batch_id}/{len(dataloader)}] | D_Loss: {float(loss_D):.4f} | G_Loss: {float(loss_G):.4f}")
            
    # 记录本轮平均 Loss
    d_losses.append(epoch_d_loss / batch_count)
    g_losses.append(epoch_g_loss / batch_count)
    
    epoch_time = time.time() - start_time
    print(f"=====> Epoch [{epoch+1}/{NUM_EPOCHS}] 完成！耗时: {epoch_time:.2f}s | 平均 D_Loss: {d_losses[-1]:.4f} | 平均 G_Loss: {g_losses[-1]:.4f}")
    
    # 定时保存图像
    current_epoch_num = epoch + 1
    if current_epoch_num in [1, 10, 50, 100, 200]:
        netG.eval()
        with paddle.no_grad():
            gen_imgs = netG(fixed_noise)
            save_generated_images(gen_imgs, current_epoch_num)
        netG.train()
        
# 【Kaggle 优化3】自动绘制并保存Loss 曲线图
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_losses, label="G Loss")
plt.plot(d_losses, label="D Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/loss_curve.png")
plt.close()
print(f"Loss 曲线图已绘制并保存至: {OUTPUT_DIR}/loss_curve.png")

print("200轮迭代全部运行结束。")
