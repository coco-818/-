# 基于PaddlePaddle的图像生成课设

## 1. 项目结构说明
本项目采用结构化、模块化的设计，源码封装在单个完整脚本中以确保一键运行的高可靠性，逻辑上划分为以下8大核心模块：
1. 开发环境配置与依赖库导入（集成 Kaggle 目录自适应优化）
2. 生成器模型定义 (Generator) —— 4层分数步长转置卷积架构
3. 判别器模型定义 (Discriminator) —— 4层全卷积二分类架构
4. 辅助函数定义：中间结果网格图像可视化保存（4x4网格输出）
5. 数据集加载与双极性预处理 (DataLoader) —— 自动拉取 CIFAR-10 训练集
6. 模型实例化、损失函数与独立 Adam 优化器配置
7. 核心对抗训练脚本（200轮双人博弈循环主体）
8. 训练收敛指标可视化（自动绘制并导出 Loss 对抗曲线图）

## 2. 运行环境依赖
本项目推荐在配备 GPU 算力加速的 Linux / Windows 环境下运行（已在 Kaggle T4 双卡平台上测试通过）：
* Python 3.12+
* PaddlePaddle-GPU == 2.6.2
* NumPy
* Matplotlib

环境一键安装命令：
pip install paddlepaddle-gpu matplotlib numpy

## 3. 项目运行方法
1. 解压项目压缩包，进入包含 `main.py` 的根目录。
2. 打开终端（Terminal）或命令行窗口，执行以下命令直接启动训练：
   python main.py

## 4. 运行行为与预期输出
* **数据集处理**：首次运行脚本时，程序会自动通过飞桨高阶内置 API 从网络安全拉取标准的 CIFAR-10 数据集并建立本地缓存（无需手动下载数据）。
* **训练日志**：程序开始迭代 200 个 Epoch（轮次），并在终端每隔 100 个 Batch 实时打印当前的判别器损失 (D_Loss) 与生成器损失 (G_Loss)。
* **图像与曲线导出**：训练开始后，程序会在当前运行目录下自动创建 `/gan_results` 文件夹，并在以下关键对抗阶段，使用固定隐向量自动生成并保存 4*4 的 16 张网格伪造图：
  * 第 1 阶段（Epoch 1）: `epoch_1_generated.png` (混沌噪声期)
  * 第 10 阶段（Epoch 10）: `epoch_10_generated.png` (基调摸索期)
  * 第 50 阶段（Epoch 50）: `epoch_50_generated.png` (结构收敛期)
  * 第 100 阶段（Epoch 100）: `epoch_100_generated.png` (半程精细期)
  * 第 200 阶段（Epoch 200）: `epoch_200_generated.png` (纳什均衡最终输出)
* **大作业图表要求**：在全部 200 轮迭代完毕后，脚本会自动绘制 G/D 双方的对抗损失演变轨迹，并最终导出为 `loss_curve.png`（Loss 曲线图）。