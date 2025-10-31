import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageDraw, ImageFont
import hashlib
# 坐标系转换函数参考：
# https://github.com/QwenLM/Qwen3-VL/issues/1706
# https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/tools/process_bbox.ipynb
# 推理代码参考：
# https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct

def convert_from_qwen3vl_format(bbox_q, h, w, mode="round", clamp_to_pixel=True, inclusive_max=False):
    """
    将 Qwen2VL 归一化坐标 [x1,y1,x2,y2] (范围 0..1000) 还原为像素坐标。
    参数:
      bbox_q: 归一化框 [x1n, y1n, x2n, y2n]
      h, w:   图像高、宽（像素）
      mode:   像素取整方式 'round' | 'floor' | 'ceil'
      clamp_to_pixel:
              True 时做边界裁剪
      inclusive_max:
              False -> 像素坐标最大为 w-1/h-1（像素索引闭区间）
              True  -> 最大为 w/h（右/下边界可等于尺寸，常用于右下“开区间”表示）
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

# ===================== Utils =====================

def _label_color(label: str) -> Tuple[int, int, int]:
    """
    根据标签名稳定地生成一种颜色（RGB）。
    """
    h = hashlib.md5(label.encode("utf-8")).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    boost = 80
    return (min(r + boost, 255), min(g + boost, 255), min(b + boost, 255))


def visualize_detections(
    image_path: str,
    detections: List[Dict[str, Any]],
    save_path: str
) -> None:
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
        # PIL>=8 有 textbbox；老版本可用 textsize 兜底
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
    从文本里提取第一段合法 JSON。
    - 先尝试直接 loads
    - 再优先提取最外层 { ... }，失败再尝试 [ ... ]
    - 自动剥离 ```json ... ``` 包裹
    """
    s = s.strip()
    if s.startswith("```"):
        # 剥掉三引号包裹（```json 或 ```）
        s = s.lstrip("`")
        # 一些模型会在第一行给出 'json' 语言标记
        if s.startswith("json"):
            s = s[4:]
        s = s.rstrip("`").strip()

    # 1) 直接尝试整体 JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) 优先尝试对象 {...}
    m_obj = re.search(r"\{[\s\S]*\}", s)
    if m_obj:
        try:
            return json.loads(m_obj.group(0))
        except Exception:
            pass

    # 3) 退回数组 [...]
    m_arr = re.search(r"\[[\s\S]*\]", s)
    if m_arr:
        return json.loads(m_arr.group(0))

    # 4) 兜底：抛错，方便定位
    raise ValueError("无法从输出中提取合法 JSON。")


# ===================== Main =====================

def main():
    local_model_path = "/home/sensing_test/Jiang_Zheng/models/Qwen3-VL-8B-Instruct"
    # local_model_path = "/origin_models/Qwen/Qwen3-VL/Qwen3-VL-8B-Instruct"
    ref_img_path = Path("/home/sensing_test/Jiang_Zheng/test_data/test10_ref.png")
    tag_img_path = Path("/home/sensing_test/Jiang_Zheng/test_data/test10_2.png")

    print(f"Loading model from local path: {local_model_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        local_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    print("Loading processor from local path...")
    processor = AutoProcessor.from_pretrained(local_model_path)

    text_prompt = """
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

    # FIX: 把 Path 转为 str，避免 qwen_vl_utils 内部 .startswith 报错
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "第一张图：目标物体"},
                {"type": "image", "image": str(ref_img_path)},   # <-- FIX
                {"type": "text", "text": "第二张图：待搜索同类物体"},
                {"type": "image", "image": str(tag_img_path)},   # <-- FIX
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    print("Preparing inputs for inference...")
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    print("Generating response...")
    generated_ids = model.generate(**inputs, max_new_tokens=4096, temperature=0.1)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    raw_output = output_texts[0]
    json_candidate = extract_first_json(raw_output)

    print("\n--- Model Output (raw) ---")
    print(json_candidate)

    try:
        result_obj = json_candidate
    except json.JSONDecodeError:
        # 再做一次宽松兜底：去掉首尾非 JSON 字符
        trimmed = json_candidate.strip()
        # 简单修剪：如果结尾有多余的逗号之类，可以尝试手动修补（可按需扩展）
        raise  # 直接抛出更明确的错误更保险

    raw_results = result_obj.get("results", [])
    if not isinstance(raw_results, list):
        print("[WARN] 返回 JSON 中未找到 `results` 列表，原始内容：")
        print(raw_output)
        sys.exit(2)

    with Image.open(str(tag_img_path)) as _im:
        W, H = _im.size

    final_results: List[Dict[str, Any]] = []
    for det in raw_results:
        if not isinstance(det, dict):
            continue
        label = det.get("label", "obj")
        score = det.get("score", None)
        # 兼容命名：bbox_qwen1000 / bbox / bbox_2d
        bbox_q = det.get("bbox")
        if not (isinstance(bbox_q, list) and len(bbox_q) == 4):
            continue
        print(f"Converting bbox : {bbox_q},height={H},width={W}")
        bbox_px = convert_from_qwen3vl_format(bbox_q, H, W)
        if not bbox_px:
            continue
        out_det = {
            "label": label,
            "bbox_qwen1000": [float(b) for b in bbox_q],  # 保留原生坐标
            "bbox_2d": bbox_px                            # 原图像素坐标
        }
        if score is not None:
            try:
                out_det["score"] = float(score)
            except Exception:
                pass
        final_results.append(out_det)

    # 输出文件路径
    out_img = str(tag_img_path.with_name(tag_img_path.stem + "_viz.jpg"))
    out_json = str(tag_img_path.with_name(tag_img_path.stem + "_results.json"))

    # 可视化并保存（用像素坐标）
    visualize_detections(str(tag_img_path), final_results, out_img)

    # 保存结构化 JSON（两套坐标都保留，方便溯源/对齐）
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"results": final_results}, f, ensure_ascii=False, indent=2)

    print(f"[OK] 可视化完成：{out_img}")
    print(f"[OK] 结果 JSON 保存：{out_json}")


if __name__ == "__main__":
    main()
