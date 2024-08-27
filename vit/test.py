import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from transformers import ViTModel


datapath='/root/nas-public-linkdata/Data'
model_path='/root/nas-public-linkdata/BDMatch/vit'
# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT 需要输入大小为 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载 CIFAR-10 数据集
train_dataset = datasets.CIFAR10(root=datapath, train=False, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)

# 加载预训练的 ViT Base 模型
vit_model = ViTModel.from_pretrained(model_path).to(device)

# 冻结 ViT 模型参数（如果只想提取特征而不进行训练）
for param in vit_model.parameters():
    param.requires_grad = False

# 提取特征
all_features = []
all_labels = []

vit_model.eval()  # 设置模型为评估模式
with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)

        # 提取 ViT 特征
        outputs = vit_model(pixel_values=images)
        features = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] token 对应的特征

        all_features.append(features.cpu())
        all_labels.append(labels)
        break

# 将所有特征和标签拼接起来
all_features = torch.cat(all_features, dim=0)
all_labels = torch.cat(all_labels, dim=0)

print(f'Extracted features shape: {all_features.shape}')
print(f'Corresponding labels shape: {all_labels.shape}')
