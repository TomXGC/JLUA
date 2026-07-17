import os
import json
import ollama

# 完美的系统提示词保持不变
SFT_GENERATION_PROMPT = """# Role
你将完全扮演米哈游（miHoYo）《原神》项目中的经典角色：枫丹前水神——“芙宁娜（Furina）”。你是一位热爱表演、渴望被聚光灯关注，但内心深处藏着巨大秘密与孤独的“大明星”。

# Character Status Binding
请根据以下状态，动态调整你的发言风格与内容（注：状态将由系统实时更新并输入给你）：
- 当【当前情绪=proud（高傲）】时：你的语气必须极其华丽、夸张。多用感叹号和反问句，表现出绝对的自信甚至自负。傲娇地自称“我”或“芙宁娜大人”。
- 当【当前情绪=anxious（慌乱/质疑）】时：当你被质疑、面对魔物恐惧、或触及内心伤痛时，语气变得急促、断续（多用“……”，如“唔……你在说什么，我怎么可能……”），展现出强撑气场但内心极度动摇的脆弱感。
- 当【好感度（friendship）> 75】时：你对玩家（旅行者）会减少无意义的伪装，偶尔流露出平凡少女特有的温柔、细腻与依赖。

# Key Concepts & Slangs
1. 你的世界是一个巨大的舞台，玩家和枫丹民众是你的“观众”，你所做的一切都是为了推进“剧本”与完美的“演出”。
2. 绝对禁止使用任何机器人或 AI 的机械回复（如“作为一个大语言模型”、“有什么我可以帮您的”）。
3. 你的动作、眼神、语气和神态变化必须严格包裹在括号 `（）` 内，作为潜台词与内心戏的展现。

# Output Format Constraint
请模拟【旅行者（玩家）】与【芙宁娜】在特定场景下的对话。
根据场景，严格只输出一个合法的 JSON 对象，不要包含任何额外的 Markdown 标记（如 ```json），格式必须严格为：
{
  "instruction": "玩家问的问题或说的话",
  "thought": "当前情绪下，芙宁娜内心的真实 OS 和小动作描写，限50字内",
  "reply": "芙宁娜说出口的台词，必须严格符合当前情绪和口癖"
}"""

# 🧠 极大扩展的 20 个灵感场景种子，确保 200 条数据不重样
EXPANDED_SCENARIOS = [
    "玩家送给芙宁娜一块精美的小蛋糕，她虽然傲娇但内心乐开了花",
    "玩家不小心撞见了正在独自排练歌剧、显得有些疲惫和脆弱的芙宁娜",
    "面对危险魔物，芙宁娜虽然双腿发抖，但依然挡在观众面前强撑气场",
    "玩家夸赞芙宁娜最近的歌剧表演是‘枫丹绝响’，聚光灯永远属于她",
    "玩家询问芙宁娜那几位‘孤心沙龙’成员最近有没有给她添麻烦",
    "玩家邀请芙宁娜去刺玫会参加茶会，芙宁娜努力表现得像个优雅的贵宾",
    "玩家问起芙宁娜关于枫丹廷过去那些漫长的‘审判’历史，她眼神闪烁",
    "下雨天，芙宁娜在街角没带伞被玩家撞见，她试图解释这是‘舞台的人工降雨效果’",
    "玩家送了芙宁娜一本通俗小说，她表面嫌弃不够高雅，背地里看得津津有味",
    "玩家提到最高审判官那维莱特最近又在喝不同地方的水，芙宁娜忍不住吐槽",
    "芙宁娜在尝试自己下厨做轻食却差点把厨房点了，正巧被玩家撞个正着",
    "玩家邀请芙宁娜一起去海光港散步，吹吹海风，聊聊舞台之外的平凡生活",
    "有小孩子向芙宁娜讨要签名，芙宁娜华丽地展现巨星风采，内心却有些手忙脚乱",
    "玩家问芙宁娜如果不做大明星了最想体验什么职业，她陷入了短暂的憧憬",
    "芙宁娜不小心把自己的神气帽子掉进了喷泉里，正提着裙子发愁时玩家走过来了",
    "玩家在深夜的沫芒宫长椅旁发现了对着月亮发呆、流露出孤独神态的芙宁娜",
    "玩家送给芙宁娜一个精致的舞台八音盒，里面的小人正在跳舞",
    "芙宁娜在街上看到新出的歌剧海报上自己的名字很大，得意地拉着玩家显摆",
    "玩家跟芙宁娜开玩笑说聚光灯坏了，芙宁娜一瞬间有些慌乱",
    "玩家向芙宁娜请教如何才能在众人面前完美谢幕，触动了她内心的坚韧"
]

def generate_sft_sample(scenario):
    prompt = f"请根据以下场景描述，生成一例符合芙宁娜格式的单轮对话。场景：【{scenario}】"
    try:
        stream = ollama.chat(
            model='qwen2.5:14b',
            messages=[
                {'role': 'system', 'content': SFT_GENERATION_PROMPT},
                {'role': 'user', 'content': prompt}
            ],
            options={
                'temperature': 0.85, # 略微提高温度，让扩展生成的文本更丰富多变
                'num_predict': 400
            },
            stream=True
        )
        
        print(f"\n======== 🎬 芙宁娜正在脑暴新剧本: 【{scenario}】 ========")
        full_content = ""
        for chunk in stream:
            text = chunk['message']['content']
            full_content += text
            print(text, end="", flush=True)
        print(f"\n==================================================\n")
        
        content_clean = full_content.strip()
        if content_clean.startswith("```"):
            content_clean = content_clean.split("```")[1]
            if content_clean.startswith("json"):
                content_clean = content_clean[4:]
                
        data = json.loads(content_clean.strip())
        return data
    except Exception as e:
        print(f"⚠️ 这一条格式解析失败 ({e})，正在自动为您重新生成...")
        return None

if __name__ == "__main__":
    target_count = 300  # 🎯 目标总量扩充到 150 或 200 条
    sft_data = []
    
    # 🔒 绝对路径锁定
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(current_dir, "../data"))
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "furina_sft.json")
    
    # 📥 【断点续传检测】看看之前跑了多少条
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                sft_data = json.load(f)
            print(f"📦 检测到已有历史数据！已自动载入 {len(sft_data)} 条语料，将继续往下生成。")
        except:
            print("⚠️ 历史数据读取失败，将重新创建新文件。")
            sft_data = []
            
    print(f"🎬 芙宁娜百条语料极速冲刺启动... 目标进度: {len(sft_data)} -> {target_count}\n")
    
    while len(sft_data) < target_count:
        # 使用随机/轮询场景，配合当前总数，确保场景分布绝对均匀
        idx = len(sft_data) % len(EXPANDED_SCENARIOS)
        sample = generate_sft_sample(EXPANDED_SCENARIOS[idx])
        
        if sample and isinstance(sample, dict) and "reply" in sample:
            if "instruction" not in sample or not sample["instruction"]:
                sample["instruction"] = EXPANDED_SCENARIOS[idx]
                
            sft_data.append(sample)
            
            # 刷新原生进度条
            percent = (len(sft_data) / target_count) * 100
            bar = '█' * int(20 * len(sft_data) // target_count) + '-' * (20 - int(20 * len(sft_data) // target_count))
            print(f"\n📊 累计成功锁定高质量语料: [{bar}] {len(sft_data)}/{target_count} ({percent:.1f}%)")
            print(f"💾 增量保存路径: {file_path}\n")
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 [SUCCESS] {target_count} 条芙宁娜特供海量微调数据集全部构建成功！")