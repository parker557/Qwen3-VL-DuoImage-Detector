import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

local_model_path = "/home/sensing_test/Jiang_Zheng/models/Qwen2.5-VL-7B-Instruct"

print(f"Loading model from local path: {local_model_path}...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    local_model_path,
    dtype=torch.bfloat16, 
    device_map="auto",
    attn_implementation="flash_attention_2"
)

print("Loading processor from local path...")
processor = AutoProcessor.from_pretrained(local_model_path)

# 修复后的消息格式 - 使用正确的字符串格式
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text", 
                "text": "这是两张图片。第一张图片（pic_apple_1.jpg）展示了一个目标物体。第二张图片（pic_apples.jpg）包含了多个物体。"
            },
            {"type": "image", "image": "/home/sensing_test/Jiang_Zheng/pic_apple_1.jpg"},
            {"type": "image", "image": "/home/sensing_test/Jiang_Zheng/pic_apples.jpg"},
            {
                "type": "text", 
                "text": '请分析第一张图片中的目标物体，然后在第二张图片中找出所有相同类型的物体，并为每个检测到的物体输出边界框，格式为: {"bbox_2d": [x,y,w,h], "label": "目标物体"}。请确保检测第二张图片中的所有同类物体，每个物体一个bbox。'
            },
        ],
    }
]

print("Preparing inputs for inference...")
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

print("Generating response...")
generated_ids = model.generate(**inputs, max_new_tokens=4096, temperature=0.3)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("\n--- Model Output ---")
print(output_text[0])
