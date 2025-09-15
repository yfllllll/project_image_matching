import gradio as gr
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
import re
import os
from PIL import Image,ImageDraw, ImageFont
from vllm import LLM, EngineArgs, SamplingParams
from typing import NamedTuple, Optional
from vllm.lora.request import LoRARequest
from dataclasses import dataclass, asdict
# 初始化模型和处理器
model_name = "/mnt/data/lyf/qwenvl-2.5-32b"  # 可替换为本地模型路径
processor = AutoProcessor.from_pretrained(model_name)

llm = LLM(model=model_name, limit_mm_per_prompt={"image": 2},tensor_parallel_size=4)
sampling_params = SamplingParams(
                temperature=0.0,max_tokens=4096)


def parse_response_boxes(response):
    """
    从响应文本中解析检测框坐标并按比例还原，并且支持多个类别。
    """
    # 创建一个字典，键是类别，值是框

    # 正则表达式修改为支持多类别的解析，输出格式：{ "bbox_2d": [x1, y1, x2, y2], "label": "{category_name}" }
    box_pattern = r'"bbox_2d": \[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\],\s*"label":\s*"(.*?)"'
    #r'"bbox_2d": \[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\],\s*"label":\s*"(.*?)"(?:,\s*"sub lable":\s*"(.*?)")?'
    #r'"bbox_2d": \[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\],\s*"label":\s*"(.*?)"'

    matches = re.findall(box_pattern, response)
    '''potential_lines = []
    for line in response[0].split('\n'):
        if '{"bbox_2d":' in line:
            potential_lines.append(line)

    # 调整正则表达式，使其更具包容性
    box_pattern = r'\{"bbox_2d":\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\],\s*"label":\s*"([^"]+)"\}'

    matches = []
    for line in potential_lines:
        match = re.search(box_pattern, line)
        if match:
            matches.append(match.groups())'''
    # 解析每一个框
    category_boxes = {}
    for match in matches:
        x1, y1, x2, y2, class_name = match
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        # 检查类别是否已经在字典中，如果不在则创建一个空列表
        #if sub_classname:
        #    class_name = class_name + '/' + sub_classname
        if class_name not in category_boxes:
            category_boxes[class_name] = []
        # 如果类别在指定的类别名列表中，则将框加入对应的类别
        category_boxes[class_name].append([x1, y1, x2, y2])

    # 生成格式化后的检测框列表
    boxes = []
    for category, box_list in category_boxes.items():
        for box in box_list:
            boxes.append((category, box))
    return boxes
   
    
def draw_boxes_on_image(image, boxes):
    """在图像上绘制检测框和标签"""
    draw = ImageDraw.Draw(image)
    
    # 尝试加载字体，失败则使用默认字体
    try:
        font = ImageFont.truetype("./wqy-microhei.ttc", 20)
    except IOError:
        font = ImageFont.load_default()
    
    for label, (x1, y1, x2, y2) in boxes:
        # 绘制矩形框（红色，宽度5）
        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        
        # 绘制标签背景
        text_width = font.getlength(label)
        draw.rectangle([x1, y1-25, x1+text_width+10, y1], fill="red")
        
        # 绘制标签文本（白色）
        draw.text((x1+5, y1-25), label, fill="white", font=font)
    
    return image

def generate_reply(images, prompt):
    """处理多图输入并生成模型输入"""
    
    image_data = [Image.open(img_path) for img_path in images]
    
    # 构建多模态对话结构
    # 构建消息格式
    txt = "如果需要返回矩形框，返回格式应为:{{\"bbox_2d\": [x1,y1,x2,y2],\"label\":\"XXX\"}}，其中XXX为目标名称。"
    prompt = f"{prompt} {txt}"
    content = []

    for idx, img in enumerate(image_data, start=1):
        # 添加图像描述文本
        content.append({"type": "text", "text": f"Image {idx}:"})
        content.append({"type": "image", "image": image_data[idx - 1]})
    # 添加文本提示
    content.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    
    # 处理视觉信息
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    outputs = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image_data},
            },
            sampling_params=sampling_params,
            )
    generated_text = outputs[0].outputs[0].text
    print(generated_text)

    # 处理每张图像的检测框
    result_images = []
    boxes = parse_response_boxes(generated_text)
    print(boxes)
    for idx, img in enumerate(image_data):
        # 解析当前图像的检测框

        
        if boxes:
            # 在图像上绘制检测框
            result_img = draw_boxes_on_image(img.copy(), boxes)
            result_images.append(result_img)
        else:
            result_images.append(img)
    
    return result_images, generated_text




