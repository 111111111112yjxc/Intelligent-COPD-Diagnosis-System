import sys
import os
import traceback
import numpy as np
import torch
import torch.nn as nn
import SimpleITK as sitk
import nibabel as nib
from scipy.ndimage import zoom
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar,
    QGroupBox, QFrame, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

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
        self.dropout = nn.Dropout(0.5)
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

class Config:
    target_size = (64, 64, 64)
    num_center_slices = 100
    lung_window = (-1000, 500)
    display_window_center = -700
    display_window_width = 1500
    binary_model_path = r"C:\Users\86187\best_resnet18_binary.pth"
    multi_model_path  = r"C:\Users\86187\best_resnet18_multi.pth"
    class_names = ['没患病', '轻度', '中度', '重度']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

binary_model = None
multi_model = None

def load_models():
    global binary_model, multi_model
    try:
        if not os.path.exists(Config.binary_model_path):
            print(f"错误：二分类模型文件不存在 - {Config.binary_model_path}")
            return False
        binary_model = resnet18_3d(num_classes=1).to(device)
        state_dict = torch.load(Config.binary_model_path, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        binary_model.load_state_dict(new_state_dict, strict=True)
        binary_model.eval()
        print("二分类模型加载成功")
    except Exception as e:
        print(f"二分类模型加载失败: {e}")
        traceback.print_exc()
        return False

    try:
        if not os.path.exists(Config.multi_model_path):
            print(f"错误：多分类模型文件不存在 - {Config.multi_model_path}")
            return False
        multi_model = resnet18_3d(num_classes=4).to(device)
        state_dict = torch.load(Config.multi_model_path, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        multi_model.load_state_dict(new_state_dict, strict=True)
        multi_model.eval()
        print("多分类模型加载成功")
    except Exception as e:
        print(f"多分类模型加载失败: {e}")
        traceback.print_exc()
        return False
    return True

class COPDPreprocessor:
    def __init__(self, dicom_folder):
        self.dicom_folder = dicom_folder
        self.original_image = None
        self.display_volume = None
        self.input_tensor = None

    def load_dicom_series(self):
        import pydicom
        from collections import Counter
        all_files = [os.path.join(self.dicom_folder, f) for f in os.listdir(self.dicom_folder)
                     if os.path.isfile(os.path.join(self.dicom_folder, f))]
        candidates = []
        for f in all_files:
            try:
                dcm = pydicom.dcmread(f, force=False)
                if 'PixelData' not in dcm:
                    continue
                if not hasattr(dcm, 'ImageOrientationPatient') or not hasattr(dcm, 'ImagePositionPatient'):
                    continue
                orient = dcm.ImageOrientationPatient
                row_dir = orient[:3]
                col_dir = orient[3:]
                if (abs(row_dir[0]) > 0.99 and abs(row_dir[1]) < 0.1 and abs(row_dir[2]) < 0.1 and
                        abs(col_dir[0]) < 0.1 and abs(col_dir[1]) > 0.99 and abs(col_dir[2]) < 0.1):
                    z = float(dcm.ImagePositionPatient[2])
                    width = dcm.Columns
                    height = dcm.Rows
                    candidates.append((z, f, width, height))
            except:
                continue
        if not candidates:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(self.dicom_folder)
            if not dicom_names:
                raise ValueError("文件夹中未找到DICOM文件")
            reader.SetFileNames(dicom_names)
            self.original_image = reader.Execute()
            self.image = self.original_image
            return self

        size_counter = Counter((w, h) for _, _, w, h in candidates)
        most_common_size = size_counter.most_common(1)[0][0]
        filtered = [(z, f) for (z, f, w, h) in candidates if (w, h) == most_common_size]
        filtered.sort(key=lambda x: x[0])
        sorted_files = [f for z, f in filtered]
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(sorted_files)
        self.original_image = reader.Execute()
        self.image = self.original_image
        return self

    def extract_center_slices(self):
        arr = sitk.GetArrayFromImage(self.image)
        z_total = arr.shape[0]
        if z_total <= Config.num_center_slices:
            return self
        start = (z_total - Config.num_center_slices) // 2
        center_arr = arr[start:start+Config.num_center_slices, :, :]
        self.image = sitk.GetImageFromArray(center_arr)
        return self

    def apply_lung_window(self):
        arr = sitk.GetArrayFromImage(self.image)
        min_val, max_val = Config.lung_window
        arr = np.clip(arr, min_val, max_val)
        arr = (arr - min_val) / (max_val - min_val)
        self.image = sitk.GetImageFromArray(arr)
        return self

    def resize_to_target(self):
        arr = sitk.GetArrayFromImage(self.image)
        if arr.shape != Config.target_size:
            zoom_factors = [t / o for t, o in zip(Config.target_size, arr.shape)]
            arr = zoom(arr, zoom_factors, order=1)
        self.display_volume = arr.copy()
        self.image = sitk.GetImageFromArray(arr)
        return self

    def standardize(self):
        arr = sitk.GetArrayFromImage(self.image)
        mean = np.mean(arr)
        std = np.std(arr) + 1e-8
        arr = (arr - mean) / std
        self.input_tensor = arr[np.newaxis, np.newaxis, ...]
        return self

    def get_inference_tensor(self):
        return torch.FloatTensor(self.input_tensor).to(device)

    def run_full_pipeline(self):
        self.load_dicom_series()
        self.extract_center_slices()
        self.apply_lung_window()
        self.resize_to_target()
        self.standardize()
        return self

class COPDWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, folder_path, parent=None):
        super().__init__(parent)
        self.folder_path = folder_path

    def run(self):
        try:
            self.progress.emit("读取DICOM序列...")
            pre = COPDPreprocessor(self.folder_path)
            pre.run_full_pipeline()

            self.progress.emit("模型推理...")
            tensor = pre.get_inference_tensor()

            with torch.no_grad():
                binary_logits = binary_model(tensor)
                binary_prob = torch.sigmoid(binary_logits).item()
                multi_logits = multi_model(tensor)
                multi_probs = torch.softmax(multi_logits, dim=1).cpu().numpy()[0]

            binary_pred = 1 if binary_prob >= 0.5 else 0
            binary_result_text = "患病" if binary_pred == 1 else "没患病"
            binary_conf = binary_prob if binary_pred == 1 else 1 - binary_prob

            result = {
                'success': True,
                'binary_pred': binary_pred,
                'binary_prob': binary_prob,
                'binary_result_text': binary_result_text,
                'binary_confidence': binary_conf,
                'multi_probs': multi_probs,
                'display_volume': pre.display_volume,
                'original_image': pre.original_image,
                'folder': self.folder_path
            }
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))
            traceback.print_exc()

class SliceRenderer:
    @staticmethod
    def get_axial_slice_from_original(image_sitk, auto_lung=True, window_center=-600, window_width=1500):
        try:
            arr = sitk.GetArrayFromImage(image_sitk)
            if arr.ndim != 3:
                return None

            if auto_lung:
                lung_mask = arr > -400
                from scipy.ndimage import label
                labeled, num_features = label(lung_mask)
                if num_features > 0:
                    sizes = np.bincount(labeled.ravel())
                    largest_label = np.argmax(sizes[1:]) + 1
                    lung_region = (labeled == largest_label)
                    z_indices = np.where(np.any(lung_region, axis=(1, 2)))[0]
                    if len(z_indices) > 0:
                        z_mid = (z_indices.min() + z_indices.max()) // 2
                    else:
                        z_mid = arr.shape[0] // 2
                else:
                    z_mid = arr.shape[0] // 2
            else:
                z_mid = arr.shape[0] // 2

            slice_2d = arr[z_mid, :, :].astype(np.float32)
            slice_2d = np.flipud(slice_2d)
            slice_2d = np.fliplr(slice_2d)

            low = window_center - window_width / 2
            high = window_center + window_width / 2
            slice_2d = np.clip(slice_2d, low, high)
            slice_2d = (slice_2d - low) / window_width * 255
            slice_2d = slice_2d.astype(np.uint8)

            h, w = slice_2d.shape
            qimage = QImage(slice_2d.tobytes(), w, h, w, QImage.Format_Grayscale8)
            return QPixmap.fromImage(qimage)
        except Exception as e:
            print(f"图像渲染错误: {e}")
            return None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("三维慢阻肺疾病智能诊断系统")
        self.setMinimumSize(1100, 850)

        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; }
            * { font-size: 11pt; }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                font-size: 12pt;
            }
            QPushButton {
                background-color: #1565C0;
                color: white;
                border: none;
                padding: 10px 16px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #0D47A1; }
            QPushButton:disabled { background-color: #B0BEC5; }
            QLabel#imageLabel {
                border: 1px solid #ddd;
                background-color: #fafafa;
            }
            QLabel#resultLabel { 
                font-size: 11pt; 
                font-weight: bold; 
                color: black; 
            }
            QLabel#confLabel { 
                font-size: 11pt; 
                color: black; 
            }
            QLabel#analysisLabel {
                font-size: 11pt;
                color: #2c3e50;
                background-color: #E3F2FD;
                padding: 10px;
                border-radius: 5px;
            }
            QProgressBar {
                border: 1px solid #B0BEC5;
                border-radius: 4px;
                text-align: center;
                min-height: 20px;
                margin: 0px;
            }
            QProgressBar::chunk {
                background-color: #1565C0;
                border-radius: 3px;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(20)

        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)

        title_layout = QHBoxLayout()
        icon_label = QLabel("🫁")
        icon_label.setStyleSheet("font-size: 32px;")
        title_label = QLabel("三维慢阻肺疾病智能诊断系统")
        title_label.setStyleSheet("font-size: 20pt; font-weight: bold; color: #0D47A1;")
        title_layout.addWidget(icon_label)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        left_layout.addLayout(title_layout)

        folder_group = QGroupBox("数据选择")
        folder_layout = QVBoxLayout()
        self.folder_label = QLabel("未选择文件夹")
        self.folder_label.setStyleSheet("background-color: white; border: 1px solid #ccc; padding: 8px;")
        self.select_btn = QPushButton("选择文件夹")
        self.select_btn.setMinimumWidth(180)
        self.select_btn.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(self.select_btn)
        folder_group.setLayout(folder_layout)
        left_layout.addWidget(folder_group)

        self.diagnose_btn = QPushButton("开始诊断")
        self.diagnose_btn.setMinimumHeight(50)
        self.diagnose_btn.setMinimumWidth(180)
        self.diagnose_btn.setStyleSheet("font-weight: bold;")
        self.diagnose_btn.clicked.connect(self.start_diagnosis)
        left_layout.addWidget(self.diagnose_btn)

        binary_group = QGroupBox("二分类诊断结果")
        binary_layout = QVBoxLayout()
        self.binary_result_label = QLabel("结果: --")
        self.binary_result_label.setObjectName("resultLabel")
        self.binary_conf_label = QLabel("置信度: --")
        self.binary_conf_label.setObjectName("confLabel")
        binary_layout.addWidget(self.binary_result_label)
        binary_layout.addWidget(self.binary_conf_label)
        binary_group.setLayout(binary_layout)
        left_layout.addWidget(binary_group)

        multi_group = QGroupBox("多分类诊断结果")
        multi_layout = QVBoxLayout()
        self.multi_probs_label = QLabel("概率: --")
        self.multi_probs_label.setWordWrap(True)
        multi_layout.addWidget(self.multi_probs_label)
        multi_group.setLayout(multi_layout)
        left_layout.addWidget(multi_group)

        analysis_group = QGroupBox("结果分析")
        analysis_layout = QVBoxLayout()
        self.analysis_label = QLabel("等待诊断...")
        self.analysis_label.setObjectName("analysisLabel")
        self.analysis_label.setWordWrap(True)
        analysis_layout.addWidget(self.analysis_label)
        self.save_btn = QPushButton("保存中间数据")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_intermediate_data)
        analysis_layout.addWidget(self.save_btn)
        analysis_group.setLayout(analysis_layout)
        left_layout.addWidget(analysis_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("background-color: #E3F2FD; padding: 8px; border-radius: 4px; font-size: 11pt;")
        left_layout.addWidget(self.status_label)

        left_layout.addStretch()

        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        self.image_label = QLabel()
        self.image_label.setObjectName("imageLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setText("诊断完成后\n将显示原始CT横断面\n（肺窗）")
        self.image_label.setStyleSheet("border: 1px solid #ccc; background-color: #fafafa; font-size: 13pt; color: #555;")
        self.image_label.setMinimumHeight(450)
        right_layout.addWidget(self.image_label)

        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 2)

        self.worker = None
        self.current_display_volume = None

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择DICOM文件夹", "")
        if folder:
            self.folder_label.setText(folder)
            self.folder_label.setToolTip(folder)

    def start_diagnosis(self):
        folder = self.folder_label.text()
        if folder == "未选择文件夹" or not os.path.isdir(folder):
            QMessageBox.warning(self, "提示", "请先选择有效的文件夹")
            return

        self.diagnose_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("处理中...")
        self.image_label.setText("正在处理，请稍候...")
        self.binary_result_label.setText("结果: --")
        self.binary_conf_label.setText("置信度: --")
        self.multi_probs_label.setText("概率: --")
        self.analysis_label.setText("等待诊断...")

        self.worker = COPDWorker(folder)
        self.worker.finished.connect(self.on_diagnosis_finished)
        self.worker.error.connect(self.on_diagnosis_error)
        self.worker.progress.connect(self.on_progress)
        self.worker.start()

    def on_progress(self, msg):
        self.status_label.setText(msg)

    def on_diagnosis_finished(self, result):
        self.worker = None
        self.diagnose_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        self.current_display_volume = result['display_volume']

        self.binary_result_label.setText(f"结果: {result['binary_pred']}")
        self.binary_conf_label.setText(f"置信度: {result['binary_confidence'] * 100:.2f}%")

        probs = result['multi_probs']
        multi_text = (f"没患病: {probs[0]:.2%}  轻度: {probs[1]:.2%}  "
                      f"中度: {probs[2]:.2%}  重度: {probs[3]:.2%}")
        self.multi_probs_label.setText(multi_text)

        analysis_text = self._generate_analysis(
            result['binary_result_text'],
            result['binary_confidence'],
            result['multi_probs']
        )
        self.analysis_label.setText(analysis_text)
        self.save_btn.setEnabled(True)

        pixmap = SliceRenderer.get_axial_slice_from_original(
            result['original_image'],
            auto_lung=True,
            window_center=-600,
            window_width=1500
        )
        if pixmap:
            scaled = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled)
            self.image_label.setText("")
        else:
            self.image_label.setText("无法生成图像")

        self.status_label.setText("诊断完成")

    def _generate_analysis(self, binary_result, binary_conf, multi_probs):
        multi_0_prob = multi_probs[0]
        probs_without_0 = multi_probs[1:]
        max_prob_without_0 = np.max(probs_without_0)
        pred_class_without_0 = np.argmax(probs_without_0) + 1
        pred_level = Config.class_names[pred_class_without_0]

        binary_is_healthy = (binary_result == "没患病")
        multi_is_healthy = (multi_0_prob > 0.5 and np.argmax(multi_probs) == 0)

        if binary_is_healthy and multi_is_healthy:
            return (f"🎯 两个模型均判断为「没患病」，结果一致。\n"
                    f"二分类置信度：{binary_conf:.2%}，多分类「没患病」概率：{multi_0_prob:.2%}。\n"
                    f"建议：无需进一步处理。")
        elif not binary_is_healthy and not multi_is_healthy:
            return (f"🎯 两个模型均判断为「患病」，结果一致。\n"
                    f"多分类模型认为最可能的等级为「{pred_level}」（概率{max_prob_without_0:.2%}）。\n"
                    f"二分类置信度：{binary_conf:.2%}。请结合临床进一步评估。")
        else:
            if binary_is_healthy and not multi_is_healthy:
                conflict_desc = f"二分类判断「没患病」（置信度{binary_conf:.2%}），而多分类判断「患病」（倾向{pred_level}，概率{max_prob_without_0:.2%}）。"
            else:
                conflict_desc = f"二分类判断「患病」（置信度{binary_conf:.2%}），而多分类判断「没患病」（概率{multi_0_prob:.2%}）。"
            if binary_conf > 0.9 or max(multi_probs) > 0.9:
                suggestion = "其中一个模型置信度较高，但仍存在矛盾，建议由医生结合CT影像进行最终判断。"
            else:
                suggestion = "两个模型置信度均不够高，建议由医生结合CT影像进行最终判断。"
            return (f"⚠️ 两个模型判断结果出现矛盾：\n{conflict_desc}\n"
                    f"{suggestion}\n"
                    f"（您可点击下方按钮保存中间数据，供医生进一步分析。）")

    def save_intermediate_data(self):
        if self.current_display_volume is None:
            QMessageBox.warning(self, "提示", "没有可保存的中间数据，请先完成诊断。")
            return
        formats = ["NumPy文件 (*.npy)", "NIfTI文件 (*.nii.gz)"]
        selected_format, _ = QFileDialog.getSaveFileName(self, "保存中间数据", "", ";;".join(formats))
        if not selected_format:
            return
        try:
            if selected_format.endswith('.npy'):
                np.save(selected_format, self.current_display_volume)
            elif selected_format.endswith('.nii.gz'):
                affine = np.eye(4)
                nifti_img = nib.Nifti1Image(self.current_display_volume, affine)
                nib.save(nifti_img, selected_format)
            else:
                np.save(selected_format, self.current_display_volume)
            QMessageBox.information(self, "保存成功", f"中间数据已保存至：\n{selected_format}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存时出错：{str(e)}")

    def on_diagnosis_error(self, err_msg):
        self.worker = None
        self.diagnose_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("处理失败")
        self.image_label.setText("处理失败，请重试")
        QMessageBox.critical(self, "错误", f"处理失败:\n{err_msg}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image_label.pixmap() and not self.image_label.pixmap().isNull():
            scaled = self.image_label.pixmap().scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()
        event.accept()

if __name__ == '__main__':
    if not load_models():
        sys.exit(1)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())