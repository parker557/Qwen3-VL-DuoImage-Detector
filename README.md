# Qwen3-VL 批量匹配检测脚本使用指南

## 概述

本脚本用于成对图片的“以图找同类”任务：基于参考图（目标物体）在待检图（包含多个同类物体）中检测所有同类物体，输出边界框并支持可视化与 CSV 导出。

## 核心功能

* **自动配对**：智能匹配 `test<序号>` 与 `test<序号>_ref` 图片文件
* **批量推理**：使用 Qwen3-VL-8B-Instruct 模型进行批量处理
* **多格式输出**：

  * 可视化图片：`output/test<序号>_result.jpg`
  * JSON 结果：`output/test<序号>_results.json`
  * CSV 汇总：`csv_output/batch_results*.csv`
* **状态记忆**：记录 CSV 路径便于追加写入
* **性能测试**：支持多 epoch 计时测试
* **场景定制提示词**：支持通过场景文件为提示词追加编号场景条目
* **模型路径缓存**：首次无缓存时交互输入；成功加载后自动缓存，后续免输入

## 目录结构

```
project/
├─ qwen3_8b_test2.py          # 主脚本
├─ input/                      # 输入目录
│  ├─ test1.jpg               # 待检图
│  ├─ test1_ref.jpg           # 参考图
│  ├─ test2.png
│  └─ test2_ref.png
├─ output/                    # 推理输出
├─ csv_output/               # CSV 输出
└─ ...
```

## 快速开始

### 环境配置

```bash
# 创建 conda 环境
conda create -n jiang-test python=3.12 -y
conda activate jiang-test

# 安装依赖
pip install torch pillow transformers openpyxl
# 安装 qwen-vl-utils（根据实际情况）
```

或使用 environment.yml：

```yaml
conda env create -f environment.yml -n jiang-test
```

### 基本使用

```bash
python qwen3_8b_test2.py \
  --input_dir input \
  --output_dir output \
  --csv_dir csv_output
```

运行时会依次询问：

* 是否追加写入 CSV（默认 y）
* 是否进行性能测试（默认 n）
* 性能测试的 epoch 次数（默认 5，当选择进行性能测试时）

> **关于模型路径（新增缓存逻辑）**
>
> * 若命令行**未指定** `--model_path`：脚本会先尝试读取缓存路径（`~/.qwen3_vl_batch_cache.json`）。
> * 若**没有缓存或缓存失效**：会提示你**交互输入**本地模型路径；如果输入为空、路径不存在或加载失败，脚本会报错退出且**不会写入缓存**。
> * 当模型**成功加载**后，路径会自动写入/刷新缓存。
> * 你也可以随时通过 `--model_path` 显式指定，成功加载后将覆盖缓存。

### 场景定制提示词（新增）

* 默认情况下，脚本使用**默认提示词**或 `--prompt` 自定义提示词。
* **仅当显式传入** `--scene_file <txt路径>` 时，才会启用**场景提示词文件**：

  * 文件应为每行一个“特殊场景描述”：

    ```
    <特殊场景描述>
    <特殊场景描述>
    <特殊场景描述>
    ```
  * 启用后，最终提示词会在说明段后**追加编号的场景条目**（1. / 2. / 3. …）。
  * 若 `--scene_file` 指定的文件**不存在或内容为空**，脚本会**报错并退出**。

示例：

```bash
# 使用默认提示词（或配合 --prompt）
python qwen3_8b_test2.py

# 使用场景提示词文件（将自动加载并生成带编号条目的提示词）
python qwen3_8b_test2.py --scene_file special_bat.txt
```

> 提示词模板片段（启用场景文件时将追加编号条目）：
>
> ```
> 根据以上两张图片。第一张图片展示了一个目标物体。第二张图片包含了多个目标物体，
> 请分析第一张图片中的目标物体，然后在第二张图片中找出所有相同类型的物体，并为每个检测到的物体输出边界框。
> 请只返回 JSON，不要多余解释、前后缀或 Markdown 代码块,且label统一使用“target_object”
>
> 1.<特殊场景描述>
> 2.<特殊场景描述>
> 3.<特殊场景描述>
>
> JSON 格式（示例值仅作结构参考）：
> {
>   "results": [
>     { "bbox": [480,365,610,440], "label": "target_object" },
>     { "bbox": [580,375,775,480], "label": "target_object" }
>   ]
> }
> ```

## 参数说明

| 参数             | 默认值                 | 说明                                             |
| -------------- | ------------------- | ---------------------------------------------- |
| `--model_path` | `""`（默认不指定）         | 本地模型路径。留空则先读缓存、无缓存时交互输入；成功加载后写入缓存。             |
| `--input_dir`  | `input`             | 输入图片目录                                         |
| `--output_dir` | `output`            | 可视化输出目录                                        |
| `--csv_dir`    | `csv_output`        | CSV 输出目录                                       |
| `--prompt`     | 默认提示词               | 未指定 `--scene_file` 时使用的文本提示词                   |
| `--scene_file` | `""`（默认不启用）         | 场景提示词 TXT 文件路径。**显式提供才启用**；若文件不存在或为空将报错退出。     |
| `--attn_impl`  | `flash_attention_2` | 注意力实现方式：`flash_attention_2` / `sdpa` / `eager` |

## 默认提示词

```
根据以上两张图片。第一张图片展示了一个目标物体。第二张图片包含了多个目标物体，
请分析第一张图片中的目标物体，然后在第二张图片中找出所有相同类型的物体，并为每个检测到的物体输出边界框。
请只返回 JSON，不要多余解释、前后缀或 Markdown 代码块,且label统一使用"target_object"。
```

期望输出格式：

```json
{
  "results": [
    { "bbox": [x1, y1, x2, y2], "label": "target_object", "score": 可选 }
  ]
}
```

## 输出文件说明

### 1. 可视化图片

* 路径：`output/test<序号>_result.jpg`
* 包含检测框和标签的可视化结果

### 2. JSON 结果文件

```json
{
  "results": [
    {
      "label": "target_object",
      "bbox_qwen1000": [480.0, 365.0, 610.0, 440.0],
      "bbox_2d": [xmin, ymin, xmax, ymax],
      "score": 0.95
    }
  ]
}
```

### 3. CSV 汇总文件

包含字段：

* `ref输入图路径`
* `ref输入图大小（H*W）`
* `输入图路径`
* `输入图大小（H*W）`
* `推理结果（只记录第一个结果）`
* `推理时间（平均时间）`
* `提示词`

## 技术实现

### 关键流程

1. **配对收集**：精确匹配 testN 与 testN_ref 文件
2. **输入构造**：文字 → 参考图 → 文字 → 待检图 → 提示词
3. **推理解析**：支持 JSON 提取和错误处理
4. **坐标转换**：从归一化坐标 (0–1000) 转换为像素坐标
5. **结果保存**：可视化、JSON、CSV 多格式输出

### 文件配对规则

* 支持格式：`.jpg/.jpeg/.png/.bmp/.webp/.tif/.tiff`
* 待检图：`test<序号>.<后缀>`
* 参考图：`test<序号>_ref.<后缀>`

## 常见问题

**Q: 提示 “发现未配对的图像” 错误**
A: 确保 `testN` 和 `testN_ref` 文件序号一致且都存在。

**Q: 模型输出不是纯 JSON**
A: 脚本会自动提取合法 JSON 部分，支持 ```json 代码块。

**Q: 显存不足或速度慢**
A: 尝试：

* 切换 `--attn_impl` 为 `sdpa` 或 `eager`
* 降低图片分辨率
* 确保正确启用 flash-attn

**Q: 场景文件生效条件**（新增）
A: 只有当显式提供 `--scene_file` 时才会启用；文件为空或不存在会报错退出；未提供时使用 `--prompt`（或默认提示词）。

**Q: 模型路径缓存规则**（新增）
A: 成功加载后才会写入 `~/.qwen3_vl_batch_cache.json`。提供了 `--model_path` 会优先使用并刷新缓存；若加载失败或输入为空/路径不存在会直接退出且不写缓存。

## 实用工具

### CSV 转 Excel 工具

```bash
# 批量转换（不递归）
python csv2xlsx.py --input-dir ./csv_output --output-dir ./xlsx_output --autowidth

# 批量转换（递归）
python csv2xlsx.py -I ./csv_output -O ./xlsx_output -R --autowidth

# 单文件转换
python csv2xlsx.py -i ./csv_output/batch_results.csv -o ./xlsx_output/batch_results.xlsx --autowidth
```

## 示例流程

```bash
# 1. 准备测试图片
mkdir -p input
cp demo_a.jpg input/test1.jpg
cp demo_a_crop.jpg input/test1_ref.jpg

# 2. 运行检测（首次未指定 --model_path 时将读取缓存或提示交互输入）
python qwen3_8b_test2.py --input_dir input --output_dir output --csv_dir csv_output

# 或显式指定模型并刷新缓存
python qwen3_8b_test2.py --model_path /models/Qwen3-VL-8B-Instruct \
  --input_dir input --output_dir output --csv_dir csv_output

# 3. 查看结果
ls output/ csv_output/
```
## 关键词
### 功能/任务类

以图找同类、以图搜图、参考图匹配、同类目标检测、目标定位、边界框标注、批量标注、可视化标注、CSV导出

image-based matching, reference-based detection, same-class detection, object localization, bounding box annotation, batch annotation, visualization, CSV export

### 提示词/交互方式

照片+提示词标注、基于提示词的检测、文本驱动检测、场景化提示词、可定制提示词、编号场景提示词

prompt-guided detection, text-prompted detection, scene prompts, customizable prompts, prompt engineering for vision

### 多模态/模型相关

多模态视觉、视觉语言模型、Qwen3-VL、Qwen3-VL-8B-Instruct、开源大模型、本地推理

multimodal vision, vision-language model, VLM, Qwen3-VL, local inference, offline inference

### 技术关键词（检索/学术风）

视觉指引、视觉对齐、视觉定位、跨图检索、特征相似度、归一化坐标、像素坐标转换、可重复试验

visual grounding, referring image, cross-image retrieval, similarity matching, normalized coords, pixel conversion, reproducibility

### 工程/流程

自动配对、批处理、性能评测、显存优化、Flash-Attn、SDPA、可配置参数、模型路径缓存、状态文件、日志

auto pairing, batch processing, benchmarking, memory optimization, flash attention, SDPA, configurable params, model path cache, state file, logging

### 典型场景（长尾短语，更容易被搜到）

“以照片输入提示词的方式标注照片”

“参考图引导在大图中找相同物体”

“批量将归一化坐标转换为像素并可视化”

“用Qwen3-VL做以图找同类并输出bbox”

“场景化提示词驱动的目标检测批处理”

“本地多模态模型的离线批量标注管线”

"Photo-to-Prompt Annotation"

"Reference-Guided Object Localization"

"Batch Normalized-to-Pixel Coordinate Conversion"

"Qwen3-VL Visual Retrieval with BBox Output"

"Context-Aware Batch Object Detection"

"Local Multimodal Batch Annotation Pipeline"

