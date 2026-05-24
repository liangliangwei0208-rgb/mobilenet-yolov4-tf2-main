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
- `VOCdevkit/`：VOC 数据集目录结构示例。

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
