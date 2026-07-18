import torch
import os
import gradio as gr
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
from modelscope import snapshot_download

# ==================== 1. 路径和环境准备 ====================
D_PROJECT_PATH = "D:/JLUA/Furina_Project"
os.environ["MODELSCOPE_CACHE"] = f"{D_PROJECT_PATH}/ModelCache"
lora_path = f"{D_PROJECT_PATH}/sft_furina_7b_lora"

print("🚀 正在预热 5070 Ti，唤醒芙宁娜的灵魂...")
base_model_path = snapshot_download("Qwen/Qwen2.5-7B-Instruct", cache_dir=f"{D_PROJECT_PATH}/ModelCache")

# ==================== 2. 4-bit 量化加载基座 ====================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto"
)

# ==================== 3. 注入 LoRA 权重 ====================
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()
print("✨ 模型加载完毕，即将启动 Web 界面！")

# ==================== 4. 核心对话与流式生成逻辑 ====================

def clean_history_html(bot_msg):
    """【历史记忆清洗器】防止模型把动态渲染的 HTML 代码学过去"""
    if "<details>" in bot_msg and "</details>" in bot_msg:
        try:
            thought_start = bot_msg.find("</summary>") + 10
            thought_end = bot_msg.find("</details>")
            thought = bot_msg[thought_start:thought_end].strip()
            reply = bot_msg[thought_end + 10:].strip()
            return f"思考：{thought}\n回复：{reply}"
        except Exception:
            return bot_msg
    return bot_msg

def extract_text(content):
    """🌟 【终极文本提取器】把所有奇奇怪怪的 Gradio 格式全部榨干成纯文本"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            elif hasattr(item, "text"):
                texts.append(item.text)
        return " ".join(texts) if texts else str(content)
    elif isinstance(content, dict) and "text" in content:
        return content["text"]
    return str(content)

def chat_with_furina(message, history):
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    
    # 拼接历史对话（榨干格式，只留纯文本）
    for item in history:
        if isinstance(item, (list, tuple)):
            user_msg = extract_text(item[0])
            bot_msg = extract_text(item[1])
            prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            clean_bot_msg = clean_history_html(bot_msg)
            prompt += f"<|im_start|>assistant\n{clean_bot_msg}<|im_end|>\n"
        else:
            role = item["role"] if isinstance(item, dict) else item.role
            raw_content = item["content"] if isinstance(item, dict) else item.content
            
            clean_content = extract_text(raw_content)
            
            if role == "user":
                prompt += f"<|im_start|>user\n{clean_content}<|im_end|>\n"
            elif role == "assistant":
                clean_bot_msg = clean_history_html(clean_content)
                prompt += f"<|im_start|>assistant\n{clean_bot_msg}<|im_end|>\n"
    
    # 拼接当前轮次的输入（同样进行榨干清洗）
    clean_message = extract_text(message)
    prompt += f"<|im_start|>user\n{clean_message}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # 实时拼装与格式化输出
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        
        if "回复：" in generated_text:
            parts = generated_text.split("回复：", 1)
            thought_part = parts[0].replace("思考：", "").strip()
            reply_part = parts[1].strip()
            
            html_output = f"<details><summary>💭 点击偷看芙宁娜的内心戏</summary>\n\n{thought_part}\n</details>\n\n**{reply_part}**"
            yield html_output
        else:
            thought_part = generated_text.replace("思考：", "").strip()
            html_output = f"<details open><summary>💭 正在读取内心...</summary>\n\n{thought_part}\n</details>"
            yield html_output

# ==================== 5. 构建现代化 Web UI ====================
demo = gr.ChatInterface(
    fn=chat_with_furina,
    title="👑 芙宁娜 AI 专属聊天室",
    description="基于 Qwen2.5-7B 微调 | 本地 RTX 5070 Ti 满血驱动",
    chatbot=gr.Chatbot(
        height=600, 
        avatar_images=(None, f"{D_PROJECT_PATH}/furina.png") 
    ),
    textbox=gr.Textbox(placeholder="跟芙宁娜大人说点什么吧...", container=False, scale=7)
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)