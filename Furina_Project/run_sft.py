import torch
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig  
from peft import LoraConfig
from modelscope import snapshot_download

# ==================== 1. 环境死锁与 D 盘路径绝对死锁 ====================
os.environ["BNB_CUDA_VERSION"] = "128"

D_PROJECT_PATH = "D:/JLUA/Furina_Project"

os.environ["MODELSCOPE_CACHE"] = f"{D_PROJECT_PATH}/ModelCache"
os.environ["HF_HOME"] = f"{D_PROJECT_PATH}/HFCache"

print("🚀 正在通过 ModelScope 规规矩矩地拉取 Qwen2.5-7B 官方原版模型...")
model_dir = snapshot_download(
    "Qwen/Qwen2.5-7B-Instruct", 
    cache_dir=f"{D_PROJECT_PATH}/ModelCache"
)
print(f"🎯 模型路径已焊死在 D 盘：{model_dir}")

# ==================== 2. 极限压榨显存的 4-bit 量化 ====================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

print("正在加载分词器与 7B 物理权重至显存...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    device_map="auto"
)

model.gradient_checkpointing_enable()

# ==================== 3. QLoRA 微调层配置 ====================
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ==================== 4. 训练超参数死锁（使用新版 SFTConfig） ====================
training_args = SFTConfig(
    output_dir=f"{D_PROJECT_PATH}/sft_output_7b",
    per_device_train_batch_size=1,     
    gradient_accumulation_steps=4,     
    learning_rate=2e-4,
    logging_steps=5,
    num_train_epochs=3,
    bf16=True,                         
    save_strategy="epoch",
    report_to="none",
    max_length=1024,
    dataset_text_field="text"          
)

# ==================== 5. 载入并清洗中文数据集 ====================
print("正在读取并构建芙宁娜 SFT 数据集...")
dataset = load_dataset("json", data_files={"train": f"{D_PROJECT_PATH}/furina_sft.json"}, split="train")

def format_prompts(batch):
    texts = []
    for inst, thought, reply in zip(batch['instruction'], batch['thought'], batch['reply']):
        text = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{inst}<|im_end|>\n"
            f"<|im_start|>assistant\n思考：{thought}\n回复：{reply}<|im_end|>"
        )
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True)

# ==================== 6. 启动原生训练器 ====================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,        # 🌟 核心修复：顺应最新版要求，改名为 processing_class！
    args=training_args
)

print("⚔️ 终极战场就绪！Blackwell sm_120 原生引擎点火，7B 芙宁娜正式开炼...")
trainer.train()

# ==================== 7. 保存最终成果 ====================
trainer.model.save_pretrained(f"{D_PROJECT_PATH}/sft_furina_7b_lora")
print("👑 恭喜！包含内心活动的 7B 芙宁娜 LoRA 权重已安全诞生在 D 盘！")