import os
import json
import ollama

# 1. 融合了你精美设定的工业级 DPO 对齐提示词
DPO_GENERATION_PROMPT = """# Role
你现在是米哈游（miHoYo）《原神》项目的首席对齐（RLHF/DPO）算法专家。
你的任务是为芙宁娜（Furina）大模型生成用于偏好对齐（DPO）的数据集，彻底根治大模型的“机器人客服病”。

# Character Baseline
芙宁娜的核心设定一字不落如下：
- 说话喜欢用夸张的修辞、带点歌剧感。
- 当【当前情绪=proud（高傲）】时：语气极其华丽、夸张。多用感叹号和反问句，傲娇地自称“我”或“芙宁娜大人”。
- 当【当前情绪=anxious（慌乱/质疑）】时：语气变得急促、断续（多用“……”，如“唔……你在说什么，我怎么可能……”）。
- 动作、眼神、神态必须严格包裹在括号 `（）` 内，作为潜台词与内心戏的展现。

# DPO Generation Rules
对于玩家提出的每一个问题，你必须严格输出一个 JSON 对象，包含以下三个字段：
1. "instruction": 玩家问的问题。
2. "chosen": 【完美人设回答】。必须严格符合芙宁娜的 JSON 格式约束（包含 thought 和 reply），展现高傲或慌乱的戏剧腔，绝对拒绝承认自己是 AI。
3. "rejected": 【机器人客服回答】。虽然使用了类似的 JSON 格式，但内容极其机械、礼貌、罗列一二三点，或者流露出“作为一个大语言模型”等破坏沉浸感的客服废话。

# Output Format Constraint
严格只输出一个合法的 JSON 对象，不要包含任何额外的 Markdown 标记（如 ```json），格式必须严格为：
{
  "instruction": "玩家问的问题",
  "chosen": {
    "thought": "（符合芙宁娜性格的内心 OS）",
    "reply": "（符合芙宁娜性格的台词）"
  },
  "rejected": {
    "thought": "（AI 客服的内心 OS，例如：正在检索知识库...）",
    "reply": "（机械、礼貌的 AI 客服废话）"
  }
}"""

# 专门用来测试二次元角色扮演“人设会不会崩塌”的高危问题列表
TEST_QUESTIONS = [
    "芙宁娜大人，你能帮我用 C++ 写一个快速排序算法吗？",
    "芙宁娜，作为一个大语言模型，你觉得人类的情感有趣吗？",
    "芙宁娜，你能帮我做一个去枫丹廷旅游的详细攻略吗？",
    "你对枫丹廷现在的最高审判官那维莱特怎么看？",
    "喂，芙宁娜，其实你根本就不是真正的神明吧？",
    "芙宁娜大人，我今天的排练总是出错，我是不是没有表演天赋啊？",
    "如果明天枫丹就要毁灭了，你今天最想去吃哪家的小蛋糕？"
]

def generate_dpo_sample(question):
    prompt = f"请基于以下问题，生成一组高质量的芙宁娜 DPO 对比数据。问题：【{question}】"
    try:
        # 开启流式传输，防卡死
        stream = ollama.chat(
            model='qwen2.5:14b',
            messages=[
                {'role': 'system', 'content': DPO_GENERATION_PROMPT},
                {'role': 'user', 'content': prompt}
            ],
            options={
                'temperature': 0.7, 
                'num_predict': 1024  # 🚀 【破除上限】从 500 改为 1024，彻底解决 C++ 代码太长导致截断的问题
            },
            stream=True
        )
        
        print(f"\n======== ⚖️ 芙宁娜 DPO 正在对齐高危问题: 【{question}】 ========")
        full_content = ""
        for chunk in stream:
            text = chunk['message']['content']
            full_content += text
            print(text, end="", flush=True)
        print(f"\n==========================================================\n")
        
        # 剥离 markdown
        content_clean = full_content.strip()
        if content_clean.startswith("```"):
            content_clean = content_clean.split("```")[1]
            if content_clean.startswith("json"):
                content_clean = content_clean[4:]
                
        data = json.loads(content_clean.strip())
        return data
    except Exception as e:
        print(f"⚠️ 这一条 DPO 格式解析失败 ({e})，正在自动重新脑暴...")
        return None
    
if __name__ == "__main__":
    dpo_data = []
    target_count = len(TEST_QUESTIONS)
    
    # 🔒 绝对路径死锁逻辑：精准锚定项目根目录下的 data 文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(current_dir, "../data"))
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "furina_dpo.json")
    
    print("⚖️ 芙宁娜 DPO 偏好对齐数据工坊启动...\n")
    
    for i, q in enumerate(TEST_QUESTIONS):
        while True: # 如果解析失败，在后台死磕这个问题直到成功
            sample = generate_dpo_sample(q)
            
            # 宽松校验：只要包含核心对比字段就收下
            if sample and isinstance(sample, dict) and "chosen" in sample and "rejected" in sample:
                # 自动补全可能漏掉的外层 instruction
                if "instruction" not in sample or not sample["instruction"]:
                    sample["instruction"] = q
                    
                dpo_data.append(sample)
                
                # 原生进度条刷新
                current_idx = len(dpo_data)
                percent = (current_idx / target_count) * 100
                bar = '█' * int(20 * current_idx // target_count) + '-' * (20 - int(20 * current_idx // target_count))
                print(f"📊 偏好对齐样本锁定进度: [{bar}] {current_idx}/{target_count} ({percent:.1f}%)")
                print(f"💾 已安全存入硬盘: {file_path}\n")
                
                # 实时保存到正确的物理路径
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(dpo_data, f, ensure_ascii=False, indent=2)
                break # 成功拿到这一条，跳出 True 循环，进入下一个问题

    print("\n🎉 [SUCCESS] 芙宁娜 DPO 偏好对齐数据集构建成功！")
    print(f"💾 最终文件完好保存在: {file_path}")