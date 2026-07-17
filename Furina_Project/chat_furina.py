import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from modelscope import snapshot_download

# ==================== 1. 路径和环境准备 ====================
D_PROJECT_PATH = "D:/JLUA/Furina_Project"
os.environ["MODELSCOPE_CACHE"] = f"{D_PROJECT_PATH}/ModelCache"
lora_path = f"{D_PROJECT_PATH}/sft_furina_7b_lora"

print("🚀 正在唤醒芙宁娜...\n")
print("第一步：加载基座躯体...")
base_model_path = snapshot_download("Qwen/Qwen2.5-7B-Instruct", cache_dir=f"{D_PROJECT_PATH}/ModelCache")

# ==================== 2. 4-bit 量化配置 ====================
# 推理时也用 4-bit 加载，确保 12G 显存毫无压力
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

# ==================== 3. 注入灵魂 (LoRA 权重) ====================
print("第二步：注入芙宁娜的灵魂 (LoRA权重)...")
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()  # 锁定模型，进入推理模式

print("\n✨ 芙宁娜已降临！(输入 '退出' 结束对话)\n" + "="*50)

# ==================== 4. 开启对话循环 ====================
while True:
    user_input = input("👤 你的输入：")
    if user_input.strip() == "退出":
        print("芙宁娜：哼，这么快就走啦？下次见！")
        break
    if not user_input.strip():
        continue

    # 严格对齐训练时的 Prompt 模版！这样她才会按格式输出“思考”和“回复”
    prompt = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 开始生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,       # 最大生成长度
            temperature=0.7,          # 创造力/傲娇程度（可微调：0.5-0.9）
            top_p=0.9,
            repetition_penalty=1.05,  # 防止无限复读
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 截取并解码刚刚生成的新内容
    input_length = inputs.input_ids.shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    print(f"\n👑 芙宁娜：\n{response}\n")
    print("-" * 50)