import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             f1_score, confusion_matrix,
                             cohen_kappa_score)
from tqdm import tqdm
from scipy.ndimage import rotate

class Config:
    train_npy_dir = r"E:\mzf-data\train_preprocessed2"
    val_npy_dir   = r"E:\mzf-data\valtest_preprocessed2"
    test_npy_dir  = r"E:\mzf-data\valtest_preprocessed2"
    train_txt = r"E:\mzf-data\data005\train.txt"
    val_txt   = r"E:\mzf-data\data005\val.txt"
    test_txt  = r"E:\mzf-data\data005\test.txt"
    batch_size = 8
    num_workers = 4
    lr = 5e-5
    epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_stop_patience = 20

class NpyDataset(Dataset):
    def __init__(self, npy_dir, txt_file, is_train=False):
        self.npy_dir = npy_dir
        self.is_train = is_train
        self.samples = []
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or '|' not in line:
                    continue
                parts = line.split('|')
                if len(parts) != 2:
                    continue
                filename = parts[0].strip()
                try:
                    label = int(parts[1].strip())
                except:
                    continue
                npy_name = filename.replace('.nii.gz', '.npy')
                npy_path = os.path.join(npy_dir, npy_name)
                if not os.path.exists(npy_path):
                    print(f"警告: npy 文件不存在 {npy_path}")
                    continue
                self.samples.append((npy_path, label))
        self.labels = np.array([s[1] for s in self.samples])
        print(f"从 {txt_file} 加载 {len(self.samples)} 个样本，标签分布: {np.bincount(self.labels)}")
        if len(self.samples) == 0:
            raise RuntimeError(f"没有有效样本: {txt_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        data = np.load(npy_path)
        if self.is_train:
            if np.random.rand() > 0.5:
                data = np.flip(data, axis=2).copy()
            if np.random.rand() > 0.5:
                data = np.flip(data, axis=3).copy()
            if label == 1 and np.random.rand() > 0.5:
                angle = np.random.uniform(-10, 10)
                data_3d = data[0]
                rotated = rotate(data_3d, angle, axes=(1,2), reshape=False, order=1)
                data = rotated[np.newaxis].copy()
        return torch.from_numpy(data).float(), torch.tensor(label, dtype=torch.float32)

class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def resnet18_3d(**kwargs):
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], **kwargs)

class COPDPipeline:
    def __init__(self, config):
        self.config = config
        self.scaler = torch.cuda.amp.GradScaler()
        self._init_data()
        self._init_model()
        self.best_metric = 0
        self.patience_counter = 0

    def _init_data(self):
        self.train_dataset = NpyDataset(self.config.train_npy_dir, self.config.train_txt, is_train=True)
        self.val_dataset   = NpyDataset(self.config.val_npy_dir,   self.config.val_txt,   is_train=False)
        self.test_dataset  = NpyDataset(self.config.test_npy_dir,  self.config.test_txt,  is_train=False)

        all_labels = np.concatenate([self.train_dataset.labels,
                                     self.val_dataset.labels,
                                     self.test_dataset.labels])
        self.num_classes = len(np.unique(all_labels))
        print(f"检测到 {self.num_classes} 个类别: {np.unique(all_labels)}")

        train_labels = self.train_dataset.labels
        class_counts = np.bincount(train_labels.astype(int), minlength=2)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[train_labels.astype(int)]
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

    def _init_model(self):
        self.model = resnet18_3d(num_classes=1).to(self.config.device).float()
        train_labels = self.train_dataset.labels
        neg_count = np.sum(train_labels == 0)
        pos_count = np.sum(train_labels == 1)
        pos_weight = torch.tensor([neg_count / (pos_count + 1e-6)]).to(self.config.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"正样本权重 pos_weight: {pos_weight.item():.4f}")

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.config.lr,
                                           weight_decay=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        preds, labels = [], []

        for inputs, targets in tqdm(self.train_loader, desc="训练"):
            inputs = inputs.to(self.config.device).float()
            targets = targets.to(self.config.device).float().view(-1, 1)

            with torch.cuda.amp.autocast(dtype=torch.float32):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            if torch.isnan(outputs).any():
                print("检测到NaN输出，跳过该批次")
                continue

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            preds.append(outputs.detach().cpu())
            labels.append(targets.cpu())

        return total_loss / len(self.train_loader), torch.cat(preds), torch.cat(labels)

    def validate(self, loader, desc="验证"):
        self.model.eval()
        total_loss = 0
        preds, labels = [], []

        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc=desc):
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device).float().view(-1, 1)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                preds.append(outputs.cpu())
                labels.append(targets.cpu())

        return total_loss / len(loader.dataset), torch.cat(preds), torch.cat(labels)

    def calculate_metrics(self, preds, labels):
        if preds.numel() == 0:
            return {}
        mask = ~torch.isnan(preds).any(dim=1)
        preds = preds[mask]
        labels = labels[mask]
        if len(preds) == 0:
            return {}

        probs = torch.sigmoid(preds).numpy().flatten()
        labels_np = labels.numpy().flatten()
        pred_labels = (probs > 0.5).astype(int)

        metrics = {
            "acc": accuracy_score(labels_np, pred_labels),
            "kappa": cohen_kappa_score(labels_np, pred_labels),
            "f1": f1_score(labels_np, pred_labels, average="binary"),
            "confusion_matrix": confusion_matrix(labels_np, pred_labels, labels=[0, 1])
        }
        try:
            metrics["auc"] = roc_auc_score(labels_np, probs)
        except Exception:
            metrics["auc"] = 0.5
        return metrics

    def run(self):
        best_val_acc = 0
        patience_counter = 0
        for epoch in range(self.config.epochs):
            train_loss, train_preds, train_labels = self.train_epoch()
            val_loss, val_preds, val_labels = self.validate(self.val_loader, desc="验证")

            train_metrics = self.calculate_metrics(train_preds, train_labels)
            val_metrics = self.calculate_metrics(val_preds, val_labels)

            if not train_metrics or not val_metrics:
                print("跳过指标计算，因存在NaN值")
                continue

            current_metric = val_metrics.get("acc", 0)
            if current_metric > best_val_acc:
                best_val_acc = current_metric
                save_path = r"C:\Users\86187\best_resnet18_binary.pth"
                torch.save(self.model.state_dict(), save_path)
                print(f"\n保存新最佳模型到 {save_path}，验证准确率: {best_val_acc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stop_patience:
                print(f"早停触发：验证准确率连续 {self.config.early_stop_patience} 个epoch未提升")
                break

            self.scheduler.step(current_metric)

            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            print(f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
            print(f"训练准确率: {train_metrics.get('acc', 0):.4f}")
            print(f"验证准确率: {val_metrics.get('acc', 0):.4f}")

        print("\n加载最佳模型，在测试集上评估...")
        self.model.load_state_dict(torch.load(r"C:\Users\86187\best_resnet18_binary.pth"))
        test_loss, test_preds, test_labels = self.validate(self.test_loader, desc="测试")
        test_metrics = self.calculate_metrics(test_preds, test_labels)
        if test_metrics:
            print(f"测试损失: {test_loss:.4f}")
            test_metric_str = " | ".join([f"{k}:{v:.4f}" for k, v in test_metrics.items() if k != "confusion_matrix"])
            print("测试指标:", test_metric_str)
            print("\n测试集混淆矩阵:")
            print(test_metrics["confusion_matrix"])
        else:
            print("测试集评估失败，无有效指标")

if __name__ == "__main__":
    config = Config()
    pipeline = COPDPipeline(config)
    pipeline.run()