#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL 性能评测脚本（不出图，输出 CSV）

功能：
- 读取一组 (ref_img, tag_img) 成对图片，运行多轮推理，记录平均耗时与首个检测结果
- 不做可视化，仅输出 CSV 汇总

CSV 表头：
ref输入图路径, 输入图大小（H*W）, ref输入图路径, 输入图大小（H*W）, 推理结果（只记录第一个结果）, 推理时间（平均时间/100epoch）, 提示词
（其中第 1、2 列为“参考图”信息，第 3、4 列为“待检图”信息）

用法示例：
python qwen3vl_perf_eval.py \
  --model-path /home/sensing_test/Jiang_Zheng/models/Qwen3-VL-8B-Instruct \
  --pairs-file /path/to/pairs.csv \
  --epochs 100 \
  --output /path/to/result.csv

pairs.csv 支持以下任一格式（自动嗅探分隔符）：
- 逗号或制表分隔：ref_img_path,tag_img_path
- 含表头或无表头均可（若含表头，要求表头包含 ref, tag 两列中的任意命名：如 ref,tag / ref_img,tag_img / reference,target 等）

计时说明：
- 统计一次完整“单次推理”耗时，包含 processor.apply_chat_template / process_vision_info / processor(...) / model.generate / decode
- 默认执行 warmup=3 次（不计入平均），随后执行 epochs 次并取均值
- 若使用 CUDA，会在起止点调用 torch.cuda.synchronize() 保证计时精确
"""

from __future__ import annotations
import argparse
import csv
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

# 进度条（若环境已安装 tqdm 则使用，否则优雅退化为普通 range）
try:  # 可选依赖
    from tqdm import tqdm as _tqdm
except Exception:  # 未安装 tqdm
    _tqdm = None


def _progress_range(total: int, desc: str = ""):
    if _tqdm is not None:
        return _tqdm(range(total), desc=desc, leave=False)
    return range(total)

# ----------------------- JSON 解析工具 -----------------------

def extract_first_json(s: str) -> Any:
    """
    从文本里提取第一段合法 JSON。
    - 先尝试直接 loads
    - 再优先提取最外层 { ... }，失败再尝试 [ ... ]
    - 自动剥离 ```json ... ``` 包裹
    """
    s = s.strip()
    if s.startswith("```"):
        s = s.lstrip("`")
        if s.startswith("json"):
            s = s[4:]
        s = s.rstrip("`").strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    m_obj = re.search(r"\{[\s\S]*\}", s)
    if m_obj:
        try:
            return json.loads(m_obj.group(0))
        except Exception:
            pass

    m_arr = re.search(r"\[[\s\S]*\]", s)
    if m_arr:
        return json.loads(m_arr.group(0))

    raise ValueError("无法从输出中提取合法 JSON。")

# ----------------------- 数据结构 -----------------------

@dataclass
class PairItem:
    ref: Path
    tag: Path

# ----------------------- I/O 工具 -----------------------

def sniff_delimiter_and_header(text: str) -> Tuple[str, bool, List[str]]:
    """自动嗅探 CSV/TSV 分隔符与是否含表头。返回 (delimiter, has_header, header_fields).
    若无表头，header_fields 以 ["ref","tag"] 占位。"""
    sample = text[:4096]
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=[",", "\t", ";", "|"])
        has_header = sniffer.has_header(sample)
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ","
        has_header = False

    f = io.StringIO(text)
    rdr = csv.reader(f, delimiter=delimiter)
    first_row = next(rdr, [])
    header_fields: List[str]
    if has_header and first_row:
        header_fields = [c.strip().lower() for c in first_row]
    else:
        header_fields = ["ref", "tag"]
    return delimiter, has_header, header_fields


def load_pairs(pairs_file: Path) -> List[PairItem]:
    text = pairs_file.read_text(encoding="utf-8")
    delimiter, has_header, headers = sniff_delimiter_and_header(text)

    # 标题名候选
    ref_keys = {"ref", "ref_img", "ref_image", "reference", "reference_img", "reference_image"}
    tag_keys = {"tag", "tag_img", "tag_image", "target", "image", "input"}

    # 读取
    pairs: List[PairItem] = []
    f = io.StringIO(text)
    rdr = csv.reader(f, delimiter=delimiter)
    if has_header:
        header = next(rdr, [])
        name2idx = {name.strip().lower(): i for i, name in enumerate(header)}
        # 匹配列索引
        ref_idx = next((name2idx[k] for k in headers if k in ref_keys and k in name2idx), None)
        tag_idx = next((name2idx[k] for k in headers if k in tag_keys and k in name2idx), None)
        if ref_idx is None or tag_idx is None:
            # 容错：尝试直接找包含关键词的列名
            for k, i in name2idx.items():
                if ref_idx is None and any(t in k for t in ["ref", "reference"]):
                    ref_idx = i
                if tag_idx is None and any(t in k for t in ["tag", "target", "image", "input"]):
                    tag_idx = i
        if ref_idx is None or tag_idx is None:
            raise ValueError("无法在表头中识别 ref/tag 列，请检查文件。")
        for row in rdr:
            if not row:
                continue
            ref = Path(row[ref_idx].strip())
            tag = Path(row[tag_idx].strip())
            pairs.append(PairItem(ref=ref, tag=tag))
    else:
        for row in rdr:
            if not row:
                continue
            if len(row) < 2:
                continue
            ref = Path(row[0].strip())
            tag = Path(row[1].strip())
            pairs.append(PairItem(ref=ref, tag=tag))

    if not pairs:
        raise ValueError("未从 pairs 文件中解析到任何样本。")
    return pairs

# ----------------------- 推理核心 -----------------------

def build_messages(ref_path: Path, tag_path: Path, text_prompt: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "第一张图：目标物体"},
                {"type": "image", "image": str(ref_path)},
                {"type": "text", "text": "第二张图：待搜索同类物体"},
                {"type": "image", "image": str(tag_path)},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]


def infer_once(model, processor, messages) -> Dict[str, Any]:
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[chat_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=4096, temperature=0.1)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    raw_output = output_texts[0]

    # 解析 JSON
    try:
        obj = extract_first_json(raw_output)
    except Exception:
        obj = {"parse_error": True, "raw": raw_output[:500]}
    return obj


def get_size_hw(path: Path) -> Tuple[int, int]:
    with Image.open(str(path)) as img:
        w, h = img.size
    return h, w

# ----------------------- 主流程 -----------------------

def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 性能评测脚本（输出 CSV，不出图）")
    parser.add_argument("--model-path", required=True, help="本地模型路径，如 /home/.../Qwen3-VL-8B-Instruct")
    parser.add_argument("--pairs-file", required=True, help="成对图片列表 CSV/TSV 文件")
    parser.add_argument("--output", required=True, help="输出 CSV 文件路径")
    parser.add_argument("--epochs", type=int, default=100, help="计时轮次（默认 100）")
    parser.add_argument("--warmup", type=int, default=3, help="预热轮次（不计入平均，默认 3）")
    parser.add_argument("--attn", default="flash_attention_2", help="transformers generate 的 attn_implementation，默认 flash_attention_2")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16","bfloat16","float32"], help="模型精度，默认 bfloat16")
    parser.add_argument("--device-map", default="auto", help="transformers device_map，默认 auto")
    parser.add_argument("--prompt", default=(
        "根据以上两张图片。第一张图片展示了一个目标物体。第二张图片包含了多个目标物体，\n"
        "请分析第一张图片中的目标物体，然后在第二张图片中找出所有相同类型的物体，并为每个检测到的物体输出边界框。\n"
        "请只返回 JSON，不要多余解释、前后缀或 Markdown 代码块,且label统一使用“target_object”。\n"
        "JSON 格式（示例值仅作结构参考）：\n"
        "{\n  \"results\": [\n    { \"bbox\": [480,365,610,440], \"label\": \"target_object\" },\n    { \"bbox\": [580,375,775,480], \"label\": \"target_object\" }\n  ]\n}\n"
    ), help="中文提示词（默认与单图推理脚本一致）")
    args = parser.parse_args()

    # dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # 加载模型/处理器
    print(f"Loading model from local path: {args.model_path} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch_dtype,
        device_map=args.device_map,
        attn_implementation=args.attn,
    )
    print("Loading processor ...")
    processor = AutoProcessor.from_pretrained(args.model_path)

    # 读取 pairs
    pairs = load_pairs(Path(args.pairs_file))
    print(f"Loaded {len(pairs)} pairs from {args.pairs_file}")

    # 准备输出 CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "ref输入图路径",
        "输入图大小（H*W）",
        "ref输入图路径",  # 这里按你的表头原样保留：第 3 列为 '待检图路径'
        "输入图大小（H*W）",
        "推理结果（只记录第一个结果）",
        "推理时间（平均时间/100epoch）",
        "提示词",
    ]

    use_cuda = torch.cuda.is_available()

    with out_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(header)

        for i, item in enumerate(pairs, 1):
            if not item.ref.exists() or not item.tag.exists():
                print(f"[WARN] 跳过第 {i} 条，路径不存在: ref={item.ref}, tag={item.tag}")
                continue

            # 记录尺寸
            ref_h, ref_w = get_size_hw(item.ref)
            tag_h, tag_w = get_size_hw(item.tag)

            # 构造消息
            messages = build_messages(item.ref, item.tag, args.prompt)

            # warmup（显示进度）
            for _ in _progress_range(max(0, args.warmup), desc=f"Warmup {i}/{len(pairs)}"):
                _ = infer_once(model, processor, messages)

            # 计时 epochs 次（显示进度）
            total_time = 0.0
            last_obj: Optional[Dict[str, Any]] = None
            for _ in _progress_range(args.epochs, desc=f"Epochs {i}/{len(pairs)}"):
                if use_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                obj = infer_once(model, processor, messages)
                if use_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                total_time += (t1 - t0)
                last_obj = obj

            avg_time = total_time / max(1, args.epochs)

            # 取第一个结果
            first_result_str = ""
            if isinstance(last_obj, dict):
                if last_obj.get("parse_error"):
                    first_result_str = f"PARSE_ERROR:{last_obj.get('raw','')[:120]}"  # 截断
                else:
                    results = last_obj.get("results")
                    if isinstance(results, list) and len(results) > 0:
                        try:
                            first_result_str = json.dumps(results[0], ensure_ascii=False, separators=(",", ":"))
                        except Exception:
                            first_result_str = str(results[0])
                    else:
                        # 若不是标准结构，则直接压缩整个对象
                        try:
                            first_result_str = json.dumps(last_obj, ensure_ascii=False, separators=(",", ":"))[:300]
                        except Exception:
                            first_result_str = str(last_obj)[:300]
            else:
                try:
                    first_result_str = json.dumps(last_obj, ensure_ascii=False)[:300]
                except Exception:
                    first_result_str = str(last_obj)[:300]

            row = [
                str(item.ref),
                f"{ref_h}*{ref_w}",
                str(item.tag),
                f"{tag_h}*{tag_w}",
                first_result_str,
                f"{avg_time:.6f}",  # 秒
                args.prompt.replace("\n", "\\n"),
            ]
            writer.writerow(row)
            print(f"[{i}/{len(pairs)}] DONE: avg={avg_time:.6f}s  ref={item.ref.name}  tag={item.tag.name}")

    print(f"[OK] 评测完成，CSV 输出：{out_path}")


if __name__ == "__main__":
    main()
