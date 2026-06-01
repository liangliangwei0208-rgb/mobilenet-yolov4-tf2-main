# mobilenet-yolov4-tf2-main

这是一个基于 TensorFlow 2 的 MobileNet-YOLOv4 目标检测项目。仓库中保留了预测、训练、评估、FPS 测试、视频检测、目录批量预测和 heatmap 可视化等入口，默认示例图片为 `img/street.jpg`。

## 项目内容

- `predict.py`：单张图片预测、视频预测、FPS 测试、目录预测和 heatmap 入口。
- `train.py`：训练入口。
- `get_map.py`：mAP 评估入口。
- `yolo.py`：预测配置和模型加载逻辑。
- `nets/`、`utils/`、`utils_coco/`：网络结构与工具函数。
- `model_data/`：类别、anchor、字体和示例权重文件。
- `img/`：测试图片。
- `VOCdevkit/`：VOC 数据集目录占位，保留目录名和说明文件；实际标注、图片和划分文件不提交到 GitHub。
- `github_gitee_sync.py`：本地、GitHub 和 Gitee 同步辅助脚本。

## 模型框架与识别原理

本项目可以理解为“轻量骨干网络 + YOLOv4 检测头”的目标检测框架。代码主要基于 TensorFlow 2 / Keras 编写，模型结构集中在 `nets/yolo.py`，预测流程集中在根目录的 `yolo.py`，后处理逻辑在 `utils/utils_bbox.py`。

### 1. 整体流程

一张图片从输入到输出检测框，大致会经过下面几步：

```text
原始图片
  -> resize/letterbox 到 416x416
  -> 主干网络提取特征
  -> SPP + PANet 融合多尺度特征
  -> YOLO Head 在 13x13、26x26、52x52 三个尺度上预测候选框
  -> 解码候选框坐标、类别和置信度
  -> 置信度过滤 + NMS 非极大值抑制
  -> 映射回原图并画框
```

新手可以先记住一句话：模型不是直接“看见物体名字”，而是把图片切成许多网格，每个网格结合多个 anchor 先验框，预测“这里有没有目标、目标框在哪里、目标属于哪个类别”。

### 2. 主干网络 Backbone

`backbone` 是特征提取网络，负责把原图转换成不同层级的特征图。本项目支持：

- `mobilenetv1`
- `mobilenetv2`
- `mobilenetv3`
- `ghostnet`
- `densenet121`
- `densenet169`
- `densenet201`

默认使用 `mobilenetv1`。MobileNet 系列的特点是使用深度可分离卷积，把普通卷积分成“逐通道卷积 + 1x1 卷积”，计算量更小，适合笔记本、低显存设备或实时检测场景。

以默认输入 `416x416` 为例，主干网络会输出三层常用特征：

```text
52x52  ：浅层特征，保留更多位置细节，适合小目标
26x26  ：中层特征，兼顾细节和语义，适合中等目标
13x13  ：深层特征，语义更强，适合大目标
```

### 3. SPP 与 PANet 特征融合

在 `nets/yolo.py` 的 `yolo_body` 中，主干网络输出特征后会进入 YOLOv4 风格的融合结构：

- SPP：使用 `13x13`、`9x9`、`5x5` 的最大池化，把不同感受野的信息拼接起来，让深层特征能看到更大范围的上下文。
- PANet：先自顶向下上采样，把深层语义传给浅层；再自底向下下采样，把浅层细节传回深层。这样小目标、大目标都能获得更合适的特征。

可以简单理解为：浅层负责“位置更准”，深层负责“语义更强”，PANet 把它们互相补充起来。

### 4. YOLO 检测头

模型最终会输出三个尺度的预测结果：

```text
13x13  ：检测大目标
26x26  ：检测中等目标
52x52  ：检测小目标
```

每个尺度上的每个网格点，会结合 3 个 anchor 先验框进行预测。每个 anchor 输出：

```text
x, y, w, h, objectness, class_probs...
```

含义如下：

- `x, y, w, h`：预测框中心点和宽高。
- `objectness`：这个框里是否有目标。
- `class_probs`：如果有目标，它分别属于各个类别的概率。

因此输出通道数是：

```text
3 * (num_classes + 5)
```

其中 `3` 是每个尺度使用 3 个 anchor，`5` 是框的位置和目标置信度信息。

### 5. 后处理：从网络输出变成可读检测框

网络输出还不是最终检测框，需要在 `utils/utils_bbox.py` 中解码：

1. 根据网格位置和 anchor，把网络预测的偏移量还原成真实框坐标。
2. 计算最终分数：`box_score = objectness * class_probability`。
3. 用 `confidence` 过滤低分框，默认是 `0.5`。
4. 用 NMS 非极大值抑制去掉重叠太多的重复框，默认 `nms_iou = 0.3`。
5. 如果输入时做了 letterbox 灰边缩放，再把框坐标修正回原始图片尺寸。

所以预测结果受两个参数影响很明显：

- `confidence` 越高，保留的框越少，误检更少但可能漏检。
- `nms_iou` 越低，重复框删除越严格；越高，可能保留更多重叠框。

### 6. 训练时模型学什么

训练入口是 `train.py`，损失函数在 `nets/yolo_training.py`。训练时模型主要学习三件事：

- 框的位置是否准确：预测框要尽量贴近标注框。
- 有没有目标：有物体的位置 objectness 应该高，背景位置应该低。
- 类别是否正确：预测类别要接近标注类别。

项目默认使用 VOC 格式数据。标注文件中的真实框会被分配到合适的尺度和 anchor 上，模型通过不断比较“预测结果”和“真实标注”的差异来更新权重。

### 7. 新手阅读代码建议

建议按这个顺序看代码：

1. `predict.py`：先看如何调用预测入口。
2. `yolo.py`：理解模型如何加载、图片如何预处理、结果如何画出来。
3. `nets/yolo.py`：理解 Backbone、SPP、PANet、YOLO Head 如何组合。
4. `utils/utils_bbox.py`：理解候选框如何解码、过滤和 NMS。
5. `train.py`：最后再看训练参数、冻结训练、解冻训练和数据增强。

如果只是想换成自己的数据集，优先关注 `classes_path`、`model_path`、`backbone`、`alpha`、`input_shape` 这几个配置，并确保训练、预测、评估三个脚本中的类别文件保持一致。

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

## 个人修复记录

### 1. 修复 TensorFlow 与 protobuf 版本不兼容

在 TensorFlow 2.2.0 环境中，如果 `protobuf` 版本过高，运行 `predict.py` 时可能在导入 TensorFlow 阶段报错：

```text
TypeError: Descriptors cannot not be created directly.
```

原因是 TensorFlow 2.2.0 与高版本 `protobuf` 不兼容。本仓库已在 `requirements.txt` 中锁定：

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

### 2. 补充 `predict.py` 图片路径说明

`predict.py` 默认 `mode = "predict"`，运行后会提示输入图片路径：

```text
Input image filename:
```

测试 `img` 文件夹下的示例图片时，需要输入：

```text
img/street.jpg
```

不要只输入 `street.jpg`，否则脚本会在当前工作目录下查找图片，导致打开失败。

### 3. 修复 GitHub 443 端口 SSH 地址识别

当前项目的 GitHub 远程地址使用：

```text
ssh://git@ssh.github.com:443/liangliangwei0208-rgb/mobilenet-yolov4-tf2-main.git
```

这是 GitHub 支持的 SSH-over-443 写法。`github_gitee_sync.py` 已支持识别该地址，不需要改回普通的 `git@github.com:owner/repo.git`。

### 4. 调整 GitHub/Gitee 同步脚本默认行为

`github_gitee_sync.py` 的 GitHub owner 默认使用 `liangliangwei0208-rgb`，GitHub 仓库名默认使用当前文件夹名。脚本默认创建或同步公开仓库；如果发现 Gitee 同名仓库已经存在但为私有，会在 `GITEE_ACCESS_TOKEN` 可用时尝试自动改为公开。

Gitee 新建仓库时可能默认打开 `master` 分支，而本项目同步分支是 `main`。脚本会在 `GITEE_ACCESS_TOKEN` 可用时尝试把 Gitee 默认分支改为 `main`；如果没有 token，需要在 Gitee 网页的仓库管理中手动把默认分支切换为 `main`。

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

如果使用自己训练的权重，需要同步修改：

- `model_path`：指向训练得到的 `.h5` 权重文件。
- `classes_path`：指向训练时使用的类别文件。
- `backbone`、`alpha`：需要与权重对应。

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

为避免仓库过大，`VOCdevkit/VOC2007` 下只提交 `Annotations`、`JPEGImages`、`ImageSets/Main` 三个目录中的 `README.md` 占位说明；实际数据文件保留在本地，不提交到 GitHub。

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
