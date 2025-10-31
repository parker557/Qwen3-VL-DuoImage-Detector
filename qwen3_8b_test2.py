# qwen3_8b_test2.py
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# 依赖：qwen-vl-utils（与你单图脚本一致）
from qwen_vl_utils import process_vision_info

import io  # 如果没有可不加，这里不强制

def _state_file(csv_dir: Path) -> Path:
    """状态文件路径：记录上一次写入的 CSV 绝对路径。"""
    return Path(csv_dir) / "last_csv.json"

def _load_last_csv_path(csv_dir: Path) -> Path | None:
    """从状态文件读取“上一次写入的 CSV 路径”。若不存在或非法，返回 None。"""
    sf = _state_file(csv_dir)
    try:
        if sf.exists():
            with open(sf, "r", encoding="utf-8") as f:
                data = json.load(f)
            p = data.get("path")
            if isinstance(p, str):
                pp = Path(p)
                return pp if pp.exists() else None
    except Exception:
        pass
    return None

def _save_last_csv_path(csv_dir: Path, csv_path: Path) -> None:
    """把本次写入的 CSV 绝对路径保存到状态文件。"""
    sf = _state_file(csv_dir)
    try:
        with open(sf, "w", encoding="utf-8") as f:
            json.dump({"path": str(csv_path.resolve())}, f, ensure_ascii=False, indent=2)
    except Exception:
        # 记录失败不影响主流程
        pass
# ===================== 工具函数 =====================
def build_base_prompt() -> str:
    return (
        "根据以上两张图片。第一张图片展示了一个目标物体。第二张图片包含了多个目标物体，\n"
        "请分析第一张图片中的目标物体，然后在第二张图片中找出所有相同类型的物体，并为每个检测到的物体输出边界框。\n"
        "请只返回 JSON，不要多余解释、前后缀或 Markdown 代码块,且label统一使用“target_object”\n\n"
        "JSON 格式（示例值仅作结构参考）：\n"
        "{\n"
        '  "results": [\n'
        '    { "bbox": [480,365,610,440], "label": "target_object" },\n'
        '    { "bbox": [580,375,775,480], "label": "target_object" }\n'
        "  ]\n"
        "}\n"
    )

def load_special_scenes(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] 场景提示词文件不存在：{p}")
        sys.exit(1)
    raw = p.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        print(f"[ERROR] 场景提示词文件为空：{p}")
        sys.exit(1)
    return lines

def build_prompt_with_scenes(special_lines: List[str]) -> str:
    header = (
        "根据以上两张图片。第一张图片展示了一个目标物体。第二张图片包含了多个目标物体，\n"
        "请分析第一张图片中的目标物体，然后在第二张图片中找出所有相同类型的物体，并为每个检测到的物体输出边界框。\n"
        "请只返回 JSON，不要多余解释、前后缀或 Markdown 代码块,且label统一使用“target_object”\n"
    )
    numbered = "\n".join(f"{i+1}.{line}" for i, line in enumerate(special_lines))
    tail = (
        "\n\nJSON 格式（示例值仅作结构参考）：\n"
        "{\n"
        '  "results": [\n'
        '    { "bbox": [480,365,610,440], "label": "target_object" },\n'
        '    { "bbox": [580,375,775,480], "label": "target_object" }\n'
        "  ]\n"
        "}\n"
    )
    return header + "\n" + numbered + tail
# ===================== 坐标转换 / 可视化 / JSON抽取 =====================

def convert_from_qwen3vl_format(
    bbox_q, h, w, mode="round", clamp_to_pixel=True, inclusive_max=False
):
    """
    将 Qwen3-VL 归一化坐标 [x1,y1,x2,y2] (范围 0..1000) 还原为像素坐标。
    """
    import math

    x1n, y1n, x2n, y2n = bbox_q
    x1 = x1n / 1000 * w
    y1 = y1n / 1000 * h
    x2 = x2n / 1000 * w
    y2 = y2n / 1000 * h

    def cast(v):
        if mode == "floor":
            return math.floor(v)
        elif mode == "ceil":
            return math.ceil(v)
        else:
            return round(v)

    x1i, y1i, x2i, y2i = map(cast, (x1, y1, x2, y2))

    if clamp_to_pixel:
        xmax = w if inclusive_max else w - 1
        ymax = h if inclusive_max else h - 1
        x1i = max(0, min(x1i, xmax))
        y1i = max(0, min(y1i, ymax))
        x2i = max(0, min(x2i, xmax))
        y2i = max(0, min(y2i, ymax))

    return [x1i, y1i, x2i, y2i]


def _label_color(label: str) -> Tuple[int, int, int]:
    """根据标签名稳定地产生一种颜色（RGB）"""
    h = hashlib.md5(label.encode("utf-8")).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    boost = 80
    return (min(r + boost, 255), min(g + boost, 255), min(b + boost, 255))


def visualize_detections(image_path: str, detections: List[Dict[str, Any]], save_path: str) -> None:
    """
    在图像上绘制 bbox 和标签。
    detections: [{"bbox_2d":[xmin,ymin,xmax,ymax], "label":"knife", "score":0.95?}, ...]
    """
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    thickness = max(2, int(min(W, H) * 0.004))   # 约 0.4% 的短边
    font_size = max(12, int(min(W, H) * 0.025))  # 约 2.5% 的短边
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        bbox = det.get("bbox_2d") or det.get("bbox") or []
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        xmin, ymin, xmax, ymax = bbox
        label = det.get("label", "obj")
        score = det.get("score", None)

        color = _label_color(str(label))
        for t in range(thickness):
            draw.rectangle([xmin - t, ymin - t, xmax + t, ymax + t], outline=color)

        text = f"{label}" if score is None else f"{label} {score:.2f}"
        try:
            bbox_text = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
        except Exception:
            tw, th = draw.textsize(text, font=font)

        bg_x1, bg_y1 = xmin, max(0, ymin - th - 4)
        bg_x2, bg_y2 = xmin + tw + 8, max(th + 4, ymin)
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=color)
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_fill = (0, 0, 0) if luminance > 160 else (255, 255, 255)
        draw.text((bg_x1 + 4, bg_y1 + 2), text, font=font, fill=text_fill)

    img.save(save_path, quality=95)


def extract_first_json(s: str):
    """
    从文本里提取第一段合法 JSON：
    - 先尝试整体 loads
    - 再尝试对象 {...}，失败再尝试数组 [...]
    - 自动剥离 ```json 包裹
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


# ===================== 批量推理 =====================

PAIR_IMG_RE = re.compile(r"^test(\d+)$", re.IGNORECASE)
PAIR_REF_RE = re.compile(r"^test(\d+)_ref$", re.IGNORECASE)
# 模型缓存文件
MODEL_CACHE_FILE = Path.home() / ".qwen3_vl_batch_cache.json"
def _load_cached_model_path() -> Path | None:
    try:
        if MODEL_CACHE_FILE.exists():
            data = json.loads(MODEL_CACHE_FILE.read_text(encoding="utf-8"))
            p = data.get("model_path")
            if isinstance(p, str):
                pp = Path(p)
                return pp if pp.exists() else None
    except Exception:
        pass
    return None

def _save_cached_model_path(p: Path) -> None:
    try:
        MODEL_CACHE_FILE.write_text(
            json.dumps({"model_path": str(p.resolve())}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        # 缓存失败不影响主流程
        pass

@dataclass
class PairItem:
    seq: str
    img_path: Path       # test<seq>.jpg
    ref_path: Path       # test<seq>_ref.jpg


def collect_pairs(input_dir: Path) -> List[PairItem]:
    # 常见图片后缀（可按需增减）
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    imgs = {}
    refs = {}

    # 遍历目录下的文件（不再只 glob *.jpg）
    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue

        stem = p.stem  # 不带后缀的文件名
        m_img = PAIR_IMG_RE.match(stem)
        m_ref = PAIR_REF_RE.match(stem)
        if m_img:
            imgs[m_img.group(1)] = p
        elif m_ref:
            refs[m_ref.group(1)] = p

    seqs = sorted(set(imgs.keys()) & set(refs.keys()), key=lambda x: int(x))
    pairs = [PairItem(seq=s, img_path=imgs[s], ref_path=refs[s]) for s in seqs]

    missing_imgs = sorted(set(refs.keys()) - set(imgs.keys()))
    missing_refs = sorted(set(imgs.keys()) - set(refs.keys()))
    if missing_imgs or missing_refs:
        lines = ["[ERROR] 发现未配对的图像，程序已终止。"]
        if missing_imgs:
            lines.append(f" - 缺少输入图：test<序列>.(任意图片后缀)   序列：{missing_imgs}")
        if missing_refs:
            lines.append(f" - 缺少参考图：test<序列>_ref.(任意图片后缀) 序列：{missing_refs}")
        print("\n".join(lines))
        sys.exit(1)

    return pairs


def build_messages(ref_img: Path, tag_img: Path, text_prompt: str):
    # Path -> str，避免内部 .startswith 报错
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "第一张图：目标物体"},
                {"type": "image", "image": str(ref_img)},
                {"type": "text", "text": "第二张图：待搜索同类物体"},
                {"type": "image", "image": str(tag_img)},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]


def run_inference_once(
    model, processor, ref_img: Path, tag_img: Path, text_prompt: str
) -> Dict[str, Any]:
    messages = build_messages(ref_img, tag_img, text_prompt)
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=4096, temperature=0.1)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    raw_output = output_texts[0]

    try:
        result_obj = extract_first_json(raw_output)
    except Exception as e:
        print(f"[ERROR] JSON 解析失败：{e}. 原始输出片段：{raw_output[:200]}...")
        result_obj = {"results": []}

    return {
        "raw": raw_output,
        "json": result_obj,
    }


def postprocess_and_save(tag_img: Path, result_obj: Dict[str, Any], out_img_path: Path, out_json_path: Path):
    # 读取原图尺寸
    with Image.open(str(tag_img)) as _im:
        W, H = _im.size

    raw_results = result_obj.get("results", [])
    final_results: List[Dict[str, Any]] = []
    for det in raw_results if isinstance(raw_results, list) else []:
        if not isinstance(det, dict):
            continue
        label = det.get("label", "obj")
        score = det.get("score", None)
        bbox_q = det.get("bbox")
        if not (isinstance(bbox_q, list) and len(bbox_q) == 4):
            continue
        bbox_px = convert_from_qwen3vl_format(bbox_q, H, W)
        out_det = {
            "label": label,
            "bbox_qwen1000": [float(b) for b in bbox_q],
            "bbox_2d": bbox_px,
        }
        if score is not None:
            try:
                out_det["score"] = float(score)
            except Exception:
                pass
        final_results.append(out_det)

    # 可视化保存
    visualize_detections(str(tag_img), final_results, str(out_img_path))

    # 保存 JSON
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump({"results": final_results}, f, ensure_ascii=False, indent=2)

    return final_results


# ===================== 主流程 =====================

DEFAULT_PROMPT = """
根据以上两张图片。第一张图片展示了一个目标物体。第二张图片包含了多个目标物体，
请分析第一张图片中的目标物体，然后在第二张图片中找出所有相同类型的物体，并为每个检测到的物体输出边界框。
请只返回 JSON，不要多余解释、前后缀或 Markdown 代码块,且label统一使用“target_object”。
JSON 格式（示例值仅作结构参考）：
{
  "results": [
    { "bbox": [480,365,610,440], "label": "target_object" },
    { "bbox": [580,375,775,480], "label": "target_object" }
  ]
}
""".strip()


def parse_args():
    ap = argparse.ArgumentParser("Qwen3-VL 批量推理")
    ap.add_argument(
        "--model_path",
        default="",   # <- 默认留空，走“缓存/首次输入”流程
        help="本地模型路径；留空则优先读取缓存，若无缓存将交互输入一次。成功加载后写入缓存。",
    )
    ap.add_argument(
        "--input_dir",
        default="input",
        help="输入目录。默认=脚本同级目录下 input/",
    )
    ap.add_argument(
        "--output_dir",
        default="output",
        help="输出目录。默认=脚本同级目录下 output/",
    )
    ap.add_argument(
        "--csv_dir",
        default="csv_output",
        help="CSV 输出目录。默认=脚本同级目录下 csv_output/",
    )
    ap.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="自定义提示词（不传则用默认）",
    )
    ap.add_argument(
        "--attn_impl",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="注意力实现方式",
    )
    ap.add_argument(
        "--scene_file",
        default="",
        help="场景提示词 TXT 文件路径。仅当显式提供时启用；若文件不存在或内容为空将报错退出。",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    input_dir = Path(args.input_dir) if args.input_dir else base_dir / "input"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "output"
    csv_dir = Path(args.csv_dir) if args.csv_dir else base_dir / "csv_output"

    # 创建目录
    for d in [input_dir, output_dir, csv_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 交互输入：是否追加 CSV / 是否性能测试
    append_csv_ans = input("是否追加写入 CSV? (y/n，默认 y): ").strip().lower() or "y"
    do_bench_ans = input("是否进行性能测试(多次Epoch)? (y/n，默认 n): ").strip().lower() or "n"
    epochs = 1
    if do_bench_ans == "y":
        try:
            epochs = int(input("请输入性能测试的 Epoch 次数(>0，默认 5): ").strip() or "5")
            if epochs <= 0:
                epochs = 5
        except Exception:
            epochs = 5

    append_csv = append_csv_ans == "y"
    do_bench = do_bench_ans == "y"

    # CSV 文件名
    if append_csv:
        # 优先使用上一次写入的 CSV
        last_path = _load_last_csv_path(csv_dir)
        if last_path is not None:
            csv_path = last_path
        else:
            # 没有历史记录或文件不存在，则创建一个固定名/或时间戳名皆可
            # 这里沿用固定名，防止到处分散多个 CSV
            csv_path = csv_dir / "batch_results.csv"
        # 只有当文件不存在或为空时才写表头
        write_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    else:
        # 不追加：新建一个带时间戳的文件
        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_path = csv_dir / f"batch_results_{ts}.csv"
        write_header = True

    # 确保记录“本次写入的 CSV 路径”，方便下次 append 时继续写
    _save_last_csv_path(csv_dir, csv_path)

    # 收集配对
    pairs = collect_pairs(input_dir)
    if not pairs:
        print(f"[INFO] 在 {input_dir} 未发现合法配对（test<序列>.jpg 与 test<序列>_ref.jpg）。")
        sys.exit(0)

    print(f"[INFO] 发现 {len(pairs)} 个配对，开始加载模型：{args.model_path}")
    # 1) 决定候选路径顺序：CLI > 缓存 > 交互输入
    candidates: List[Path] = []
    cli_path = (args.model_path or "").strip()
    if cli_path:
        candidates.append(Path(cli_path))
    else:
        cached = _load_cached_model_path()
        if cached is not None:
            candidates.append(cached)

    model = None
    processor = None
    chosen_path: Path | None = None
    load_err: Exception | None = None

    def _try_load_model(p: Path) -> Tuple[Any, Any] | None:
        # 验证基本存在性
        if not p.exists():
            print(f"[ERROR] 模型路径不存在：{p}")
            return None
        try:
            m = Qwen3VLForConditionalGeneration.from_pretrained(
                str(p),
                dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation=args.attn_impl,
            )
            proc = AutoProcessor.from_pretrained(str(p))
            return m, proc
        except Exception as e:
            print(f"[ERROR] 模型加载失败：{p}\n{repr(e)}")
            return None

    # 2) 先尝试已知候选（CLI 或 缓存）
    for cand in candidates:
        ret = _try_load_model(cand)
        if ret is not None:
            model, processor = ret
            chosen_path = cand
            break

    # 3) 若还没成功，首次使用 -> 交互输入一次
    if model is None:
        in_path = input("首次使用：请输入本地模型路径（绝对或相对路径）：").strip()
        if not in_path:
            print("[ERROR] 模型路径为空，已退出。")
            sys.exit(1)
        cand = Path(in_path)
        ret = _try_load_model(cand)
        if ret is None:
            print("[ERROR] 模型载入失败（路径为空/不存在/结构错误或权重不可用），已退出。")
            sys.exit(1)
        model, processor = ret
        chosen_path = cand

    # 4) 仅在“成功加载后”写入/刷新缓存
    if chosen_path is not None:
        _save_cached_model_path(chosen_path)
        print(f"[INFO] 模型已成功加载并缓存路径：{chosen_path}")
    scene_path = (args.scene_file or "").strip()
    if scene_path:
        # 有提供路径 -> 按你的逻辑读取与构建；空/不存在会在 load_special_scenes 内退出
        special_lines = load_special_scenes(scene_path)
        prompt_text = build_prompt_with_scenes(special_lines)
    else:
        # 未提供 -> 使用 --prompt（默认就是 DEFAULT_PROMPT）
        prompt_text = args.prompt

    # 打开 CSV
    with open(csv_path, "a", newline="", encoding="utf-8") as csv_f:
        writer = csv.writer(csv_f)
        if write_header:
            writer.writerow([
                "ref输入图路径",
                "ref输入图大小（H*W）",
                "输入图路径",
                "输入图大小（H*W）",
                "推理结果（只记录第一个结果）",
                "推理时间（平均时间）",
                "提示词",
            ])

        for item in pairs:
            out_img = output_dir / f"test{item.seq}_result.jpg"
            out_json = output_dir / f"test{item.seq}_results.json"

            # 读取尺寸
            with Image.open(str(item.ref_path)) as im_ref:
                ref_w, ref_h = im_ref.size
            with Image.open(str(item.img_path)) as im_tag:
                tag_w, tag_h = im_tag.size

            # 计时（平均）
            times: List[float] = []
            first_result_obj: Dict[str, Any] = {}
            first_final_results: List[Dict[str, Any]] = []

            repeat = epochs if do_bench else 1
            for ep in range(repeat):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = run_inference_once(model, processor, item.ref_path, item.img_path, prompt_text)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)

                if ep == 0:
                    # 仅第一次做可视化与存盘
                    first_result_obj = out["json"] if isinstance(out, dict) else {}
                    first_final_results = postprocess_and_save(
                        item.img_path, first_result_obj, out_img, out_json
                    )

            avg_time = sum(times) / len(times) if times else 0.0

            # CSV：只记录第一个结果（若有）
            first_det = first_final_results[0] if first_final_results else {}
            first_det_str = json.dumps(first_det, ensure_ascii=False)

            writer.writerow([
                str(item.ref_path),
                f"{ref_h}*{ref_w}",
                str(item.img_path),
                f"{tag_h}*{tag_w}",
                first_det_str,
                f"{avg_time:.4f}",
                prompt_text.replace("\n", "\\n"),
            ])

            print(f"[OK] 序列 {item.seq} 完成：结果图 -> {out_img.name}，平均耗时 {avg_time:.4f}s")

    print(f"[DONE] 全部完成。CSV 已写入：{csv_path}")


if __name__ == "__main__":
    main()
