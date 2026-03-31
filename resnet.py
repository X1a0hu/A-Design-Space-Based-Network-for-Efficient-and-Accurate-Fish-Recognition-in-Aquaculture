import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
import sys
from sklearn.metrics import accuracy_score, pairwise
import os

sys.path.append(str(Path(__file__).parent))

from identify.dataset import FishDataset
from identify.losses import LargeMarginCosineLoss
from identify.estimator import Estimator


def get_model(resnet_name):
    model = None
    if resnet_name == "ResNet18":
        model = models.resnet18(weights=None)
    elif resnet_name == "ResNet34":
        model = models.resnet34(weights=None)
    elif resnet_name == "ResNet50":
        model = models.resnet50(weights=None)
    elif resnet_name == "ResNet101":
        model = models.resnet101(weights=None)

    model.fc = nn.Linear(model.fc.in_features, 512)

    return model


resnet_names = ["ResNet18", "ResNet34", "ResNet50", "ResNet101"]


def summ():
    from fvcore.nn import FlopCountAnalysis
    import torchsummary

    for name in resnet_names:
        model = get_model(name)

        model.to("cuda")
        torchsummary.summary(model, input_size=(3, 224, 224))
        flops = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224).to("cuda"))
        print(f"Total GFLOPs: {flops.total() / 1e9:.4f}")


def train():
    # 数据预处理
    transform_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # 调整大小
            transforms.RandomAffine(degrees=20),
            transforms.ToTensor(),  # 转换为 Tensor
            transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1]),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # 调整大小
            transforms.ToTensor(),  # 转换为 Tensor
            transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1]),
        ]
    )

    base_dir = os.getcwd()
    data = os.path.join(base_dir, "cfg", "datasets", "fish-identify-n.yaml")

    train_dataset = FishDataset(data=data, transform=transform_train, mode="train")
    val_dataset = FishDataset(data=data, transform=transform_val, mode="val")

    # 加载数据集
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=1)

    # 类别数
    num_classes = train_dataset.nc
    print(f"类别数: {num_classes}")

    # 加载 ResNet50 预训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model().to(device)

    import torch.optim as optim

    # 交叉熵损失
    criterion = LargeMarginCosineLoss(
        embed_dim=512, num_classes=num_classes, m=0.35, s=32
    )
    criterion.to("cuda")

    # Adam 优化器
    optimizer = optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, fused=True, weight_decay=1e-4
    )

    num_epochs = 1000

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(0.9 * num_epochs), eta_min=1e-4
    )
    # scheduler = optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[50, 150], gamma=0.5
    # )
    est = Estimator(
        save_dir=Path.cwd(),
        device=device,
    )

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=True)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            # 训练
            embedding = model(images)
            loss = criterion(embedding, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        print()

        model.eval()
        results = est.estimate(model=model, dataloder=val_loader)
        print(
            f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, rank_1: {results['rank_1']}, tar_far: {results['tar_far']}"
        )


if __name__ == "__main__":
    summ()
    # train()

    # correct, total = 0, 0
    # all_labels = []
    # all_predictions = []
    # features = []

    # with torch.no_grad():
    #     for images, labels in val_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         feature, logits = model(images)
    #         features.append(feature.cpu().numpy())
    #         _, predicted = logits.max(1)

    #         correct += predicted.eq(labels).sum().item()
    #         total += labels.size(0)
    #         all_labels.extend(labels.cpu().numpy())
    #         all_predictions.extend(predicted.cpu().numpy())

    # features = np.vstack(features)
    # features = features / np.linalg.norm(features, axis=1, keepdims=True)
    # labels_list = np.hstack(all_labels)

    # similarity_matrix = pairwise.cosine_similarity(features)

    # pos_scores = []
    # neg_scores = []

    # for i in range(len(labels_list)):
    #     for j in range(i + 1, len(labels_list)):
    #         sim_score = similarity_matrix[i, j]
    #         if labels_list[i] == labels_list[j]:
    #             pos_scores.append(sim_score)
    #         else:
    #             neg_scores.append(sim_score)

    # # 设定 FAR = 1e-6，计算 TAR
    # neg_scores = np.sort(neg_scores)[::-1]  # 降序排序（FAR 阈值对应较高的负例分数）
    # far_threshold = int(len(neg_scores) * 1e-6)
    # threshold = (
    #     neg_scores[far_threshold] if far_threshold < len(neg_scores) else neg_scores[-1]
    # )

    # # 计算 TAR（大于阈值的正样本比例）
    # tar = np.mean(np.array(pos_scores) >= threshold) * 100

    # rank1_acc = accuracy_score(all_labels, all_predictions) * 100
    # print(
    #     f"Val Accuracy: {100.0 * correct / total:.2f}%, Rank-1 Accuracy: {rank1_acc:.2f}%, TAR@FAR=1e-6: {tar:.2f}%"
    # )
