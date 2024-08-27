import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from transformers import ViTModel, ViTConfig

# 定义类别和设备
selected_classes = [0, 1, 2, 3, 4, 5]  # CIFAR-10 中的前6个类别
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT 需要输入大小为 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载 CIFAR-10 数据集并选择指定类别
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


# 过滤出前6个类别的数据
def filter_dataset(dataset, classes):
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    dataset = Subset(dataset, indices)
    return dataset


train_dataset = filter_dataset(train_dataset, selected_classes)
test_dataset = filter_dataset(test_dataset, selected_classes)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# 加载预训练的 ViT Base 模型
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)

# 冻结 ViT 模型参数
for param in vit_model.parameters():
    param.requires_grad = False


# 定义一个 One-Versus-All 分类器头
class OVAHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(OVAHead, self).__init__()
        self.classifiers = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(num_classes)])

    def forward(self, x):
        logits = [classifier(x) for classifier in self.classifiers]
        return torch.cat(logits, dim=1)


# 获取 ViT 模型的特征维度
feature_dim = vit_model.config.hidden_size

# 添加 OVA 分类头
ova_head = OVAHead(input_dim=feature_dim, num_classes=len(selected_classes)).to(device)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(ova_head.parameters(), lr=1e-3)

# 训练模型
for epoch in range(10):  # 设定训练轮数
    ova_head.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 将标签转换为 One-Versus-All 格式
        labels_ova = torch.zeros(labels.size(0), len(selected_classes)).to(device)
        for i, label in enumerate(selected_classes):
            labels_ova[:, i] = (labels == label).float()

        # 提取 ViT 特征
        with torch.no_grad():
            features = vit_model(pixel_values=images).last_hidden_state[:, 0, :]

        # 通过 OVA 分类头
        outputs = ova_head(features)

        # 计算损失
        loss = criterion(outputs, labels_ova)

        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# 在测试集上评估模型
ova_head.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        labels_ova = torch.zeros(labels.size(0), len(selected_classes)).to(device)
        for i, label in enumerate(selected_classes):
            labels_ova[:, i] = (labels == label).float()

        features = vit_model(pixel_values=images).last_hidden_state[:, 0, :]
        outputs = ova_head(features)

        predicted = torch.sigmoid(outputs).round()
        correct += (predicted == labels_ova).sum().item()
        total += labels_ova.numel()

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')
