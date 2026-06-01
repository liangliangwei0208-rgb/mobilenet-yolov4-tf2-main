# mobilenet-yolov4-tf2-main

这是一个基于 TensorFlow 2 / Keras 的轻量目标检测项目。模型使用 MobileNet 系列等网络作为特征提取主干，在 YOLOv4 风格的多尺度检测结构上完成目标定位和类别识别。项目保留了单图预测、视频检测、目录预测、FPS 测试、heatmap 可视化、训练和 mAP 评估等常用入口，默认测试图片为 `img/street.jpg`。

## 项目内容

- `predict.py`：预测入口，可切换单图、视频、FPS、目录预测和 heatmap 模式。
- `train.py`：训练入口，包含冻结训练、解冻训练、优化器、学习率和数据增强配置。
- `get_map.py`：mAP 评估入口，用于训练后检查检测效果。
- `yolo.py`：预测阶段的配置、模型加载、图像预处理、结果绘制逻辑。
- `nets/`：网络结构和训练损失，包括主干网络、YOLO 检测结构和 loss。
- `utils/`：数据读取、图像处理、候选框解码、NMS、训练回调等工具代码。
- `utils_coco/`：COCO 格式相关工具。
- `model_data/`：类别文件、anchor 文件、字体文件和示例权重。
- `img/`：测试图片。
- `VOCdevkit/`：VOC 数据集目录占位，实际图片、标注和划分文件不纳入版本管理。

## 模型框架与识别原理

本项目的检测流程可以拆成四个部分：输入预处理、特征提取、特征融合和检测后处理。代码中主要对应 `yolo.py`、`nets/yolo.py`、`utils/utils_bbox.py` 和 `utils/dataloader.py`。

### 1. 输入如何进入模型

预测时，`yolo.py` 中的 `detect_image` 会先把图片整理成模型可以处理的张量：

1. 使用 `cvtColor` 将输入图像转换为 RGB，避免灰度图或其他格式导致通道数不匹配。
2. 使用 `resize_image` 将图片缩放到 `input_shape`，默认是 `416x416`。
3. 当 `letterbox_image=True` 时，图片会按比例缩放，并在空白区域填充灰边，这样可以尽量保持原始宽高比例。
4. 使用 `preprocess_input` 将像素值从 `0-255` 归一化到 `0-1`。
5. 使用 `np.expand_dims` 增加 batch 维度，最终输入形状类似 `(1, 416, 416, 3)`。

这一步不做识别，只负责把原图转换成网络可以稳定计算的格式。后面输出检测框时，如果使用了 letterbox，代码还会把灰边造成的坐标偏移修正回原图尺寸。

### 2. 主干网络如何提取特征

默认配置使用 `mobilenetv1` 作为 backbone。MobileNetV1 的核心是深度可分离卷积，它把一次普通卷积拆成两步：

- `DepthwiseConv2D`：每个输入通道单独做空间卷积，提取边缘、纹理、局部形状等空间信息。
- `1x1 Conv2D`：把不同通道的信息重新组合，形成更有表达能力的特征。

相比普通卷积，这种结构计算量更低，适合笔记本和实时检测场景。代码位置是 `nets/mobilenet_v1.py` 中的 `_depthwise_conv_block`。

以默认 `416x416` 输入为例，MobileNetV1 会逐级下采样，并取出三个有效特征层：

```text
feat1: 52x52，步长 8，位置细节较多，主要服务小目标
feat2: 26x26，步长 16，细节和语义比较均衡，主要服务中等目标
feat3: 13x13，步长 32，感受野最大，主要服务大目标
```

这里的“特征”不是原图像素，而是卷积网络提取出的高维响应。浅层特征更接近纹理和边缘，定位能力较强。深层特征经过多次卷积和下采样，包含更多类别语义，但空间分辨率更低。

项目还支持 `mobilenetv2`、`mobilenetv3`、`ghostnet`、`densenet121`、`densenet169` 和 `densenet201`。这些主干网络的输出通道数不同，但都会向后续检测结构提供三层有效特征。

### 3. 特征如何融合

`nets/yolo.py` 中的 `yolo_body` 接收 `feat1 / feat2 / feat3` 后，会构建 YOLOv4 风格的特征融合结构。

首先，最深层的 `feat3` 会进入 SPP 结构。代码中使用 `13x13`、`9x9`、`5x5` 三个最大池化分支，再和原特征拼接。这样做的目的，是让深层特征同时包含不同范围的上下文信息。对于遮挡、背景复杂或目标尺度变化较大的图片，这类上下文信息会影响最终分类和定位。

随后进入 PANet 融合过程：

- 自顶向下路径：深层特征上采样后与中层、浅层特征拼接，把强语义信息传给高分辨率特征图。
- 自底向上路径：浅层融合结果再逐级下采样，与中层、深层特征拼接，把更清楚的位置细节带回低分辨率特征图。

这一步的结果是三组融合后的特征图。它们分别保留不同尺度的信息，使模型不只依赖单一大小的特征图进行检测。

### 4. 检测头如何生成候选框

模型最终输出三层预测结果：

```text
P5_output: 13x13，偏向大目标
P4_output: 26x26，偏向中等目标
P3_output: 52x52，偏向小目标
```

每一层特征图上的每个网格点都会结合 3 个 anchor 进行预测。单个 anchor 需要输出 `num_classes + 5` 个数值：

```text
x, y, w, h, objectness, class_probs...
```

其中：

- `x, y` 表示预测框中心点相对当前网格的位置偏移。
- `w, h` 表示预测框宽高相对 anchor 的缩放。
- `objectness` 表示这个位置是否存在目标。
- `class_probs` 表示该目标属于各类别的概率。

因此每个检测层的输出通道数是：

```text
3 * (num_classes + 5)
```

如果使用 VOC 默认 20 类，输出通道就是 `3 * (20 + 5) = 75`。这一点和 `nets/yolo.py` 中 `DarknetConv2D(len(anchors_mask[i]) * (num_classes + 5), (1,1))` 的写法一致。

### 5. 网络输出如何变成真实检测框

网络原始输出还不能直接画在图片上，需要在 `utils/utils_bbox.py` 中解码。核心函数是 `get_anchors_and_decode` 和 `DecodeBox`。

解码过程主要做几件事：

1. 根据特征图尺寸生成网格坐标，例如 `13x13` 特征图会生成 13 行 13 列的网格位置。
2. 对 `x, y` 使用 `sigmoid`，让中心点落在当前网格附近，再加上网格坐标并归一化。
3. 对 `w, h` 使用 `exp`，再乘以对应 anchor 的宽高，得到预测框尺寸。
4. 对 `objectness` 和 `class_probs` 使用 `sigmoid`，得到目标置信度和类别概率。
5. 将三个尺度的预测结果展平成同一组候选框。
6. 使用 `yolo_correct_boxes` 把归一化坐标映射回原图尺寸，并处理 letterbox 灰边偏移。

最终每个候选框都会得到一个类别分数：

```text
box_score = objectness * class_probability
```

`confidence` 控制候选框是否保留，默认值是 `0.5`。分数低于阈值的框会被过滤掉。

### 6. 模型如何确定最终识别结果

经过置信度过滤后，同一个目标附近通常还会剩下多个重叠框。`DecodeBox` 会按类别执行 NMS 非极大值抑制：

1. 对每个类别分别取出超过置信度阈值的候选框。
2. 按分数从高到低排序。
3. 保留当前分数最高的框。
4. 删除与它 IoU 过高的其他框。
5. 重复这个过程，直到没有可继续处理的框。

`nms_iou` 控制“重叠过高”的判断标准，默认是 `0.3`。最终输出的是三个数组：检测框坐标、检测分数和类别编号。`yolo.py` 再根据 `voc_classes.txt` 把类别编号转换成类别名称，并把检测框画回图片。

### 7. 训练时模型在学习什么

训练阶段的数据由 `utils/dataloader.py` 处理。标注框会被转换成三层 `y_true`，形状分别对应 `13x13`、`26x26` 和 `52x52`。每个真实框会根据宽高与 anchor 的 IoU，分配给最合适的 anchor 和特征层。

训练损失在 `nets/yolo_training.py` 中计算，主要包含三部分：

- 位置损失：使用 CIoU 约束预测框和真实框的位置、大小及重叠关系。
- 置信度损失：让有目标的位置输出高 objectness，让背景位置输出低 objectness。
- 分类损失：让正样本位置预测出正确类别。

训练过程中还会使用 `ignore_mask`。当某个预测框虽然没有被分配为正样本，但它和真实框已经很接近时，不把它当成普通背景强行惩罚。这样可以减少训练时的误伤。

数据增强由 `utils/dataloader.py` 完成，包含随机缩放、裁剪、翻转、颜色扰动、Mosaic 和 MixUp。增强后的图片和标注框会一同变换，保证训练时输入和标签仍然对应。

### 8. 阅读代码建议

建议按下面顺序阅读源码：

1. `predict.py`：先看预测入口和模式切换。
2. `yolo.py`：理解模型加载、预处理、预测调用和画框。
3. `nets/mobilenet_v1.py`：理解默认主干网络怎样逐级提取特征。
4. `nets/yolo.py`：理解 SPP、PANet 和三尺度检测头。
5. `utils/utils_bbox.py`：理解候选框解码、置信度过滤和 NMS。
6. `utils/dataloader.py`：理解真实框如何分配到三层特征图。
7. `nets/yolo_training.py`：最后再看 CIoU、置信度损失和分类损失。

如果只是训练自己的数据，优先确认 `classes_path`、`model_path`、`backbone`、`alpha`、`input_shape` 和 `anchors_path`，并保证训练、预测、评估三个阶段使用同一套类别定义。

## 运行环境

建议使用 Python 3.7 环境。本项目已在如下组合中验证：

```text
Python 3.7.16
tensorflow-gpu==2.2.0
protobuf==3.20.3
```

安装依赖：

```bash
pip install -r requirements.txt
```

### TensorFlow 与 protobuf 版本

在 TensorFlow 2.2.0 环境中，如果 `protobuf` 版本过高，运行 `predict.py` 时可能在导入 TensorFlow 阶段报错：

```text
TypeError: Descriptors cannot not be created directly.
```

原因是 TensorFlow 2.2.0 与高版本 `protobuf` 不兼容。本项目已在 `requirements.txt` 中锁定：

```text
protobuf==3.20.3
```

如果本地环境已经安装了过高版本，可以执行：

```bash
pip uninstall -y protobuf
pip install protobuf==3.20.3
```

验证版本：

```bash
python -c "import tensorflow as tf; import google.protobuf; print(tf.__version__, google.protobuf.__version__)"
```

期望输出类似：

```text
2.2.0 3.20.3
```

## 快速预测

1. 激活环境并进入项目根目录。
2. 确认 `model_data/yolov4_mobilenet_v1_voc.h5` 存在。
3. 运行：

```bash
python predict.py
```

4. 出现提示后输入：

```text
img/street.jpg
```

`predict.py` 默认 `mode = "predict"`，运行后会提示输入图片路径：

```text
Input image filename:
```

测试 `img` 文件夹下的示例图片时，需要输入完整相对路径 `img/street.jpg`。如果只输入 `street.jpg`，脚本会在当前工作目录下查找图片，可能导致打开失败。

测试通过时，脚本会加载默认权重，并在命令行输出检测框信息。

## 常用配置

预测相关配置集中在 `yolo.py` 的 `_defaults` 中：

```python
"model_path"        : "model_data/yolov4_mobilenet_v1_voc.h5",
"classes_path"      : "model_data/voc_classes.txt",
"anchors_path"      : "model_data/yolo_anchors.txt",
"input_shape"       : [416, 416],
"backbone"          : "mobilenetv1",
"confidence"        : 0.5,
"nms_iou"           : 0.3,
```

如果使用自己训练的权重，需要同时修改：

- `model_path`：指向训练得到的 `.h5` 权重文件。
- `classes_path`：指向训练时使用的类别文件。
- `backbone`、`alpha`：需要与权重对应。
- `anchors_path`：如果重新聚类过 anchor，需要指向新的 anchor 文件。

## 训练说明

### VOC 格式数据

项目默认按 VOC 格式组织数据：

```text
VOCdevkit/
  VOC2007/
    Annotations/
    JPEGImages/
    ImageSets/
      Main/
```

为避免项目文件过大，`VOCdevkit/VOC2007` 下只保留 `Annotations`、`JPEGImages`、`ImageSets/Main` 三个目录中的说明文件。实际图片、标注文件和数据集划分文件保留在本地，不纳入版本管理。

训练前需要：

1. 将标注文件放入 `VOCdevkit/VOC2007/Annotations/`。
2. 将图片放入 `VOCdevkit/VOC2007/JPEGImages/`。
3. 按自己的类别修改类别文件。
4. 运行 `voc_annotation.py` 生成训练列表。
5. 运行 `train.py` 开始训练。

训练自己的数据集时，`voc_annotation.py`、`train.py`、`yolo.py`、`get_map.py` 中的类别文件路径需要保持一致。

## 评估说明

完成训练后，可运行：

```bash
python get_map.py
```

评估结果会输出到 `map_out/`。如果 mAP 为 0，优先检查：

- `classes_path` 是否与训练类别一致。
- `model_path` 是否指向正确权重。
- 标注文件与图片路径是否正确。
- 预测阶段是否能正常检测出目标。

## 其他模式

`predict.py` 中的 `mode` 可切换：

- `predict`：单张图片预测。
- `video`：视频或摄像头检测。
- `fps`：FPS 测试。
- `dir_predict`：遍历目录预测并保存结果。
- `heatmap`：生成预测热力图。

## 许可证

本项目保留 MIT 许可证文本，详见 `LICENSE`。
