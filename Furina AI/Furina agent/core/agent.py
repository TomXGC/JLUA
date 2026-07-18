import ollama
from core.npc import FurinaNPC

class FurinaAgent:
    def __init__(self,model_name = "qwen2.5:14b"):
        self.npc = FurinaNPC
        self.model_name = model_name
        self.history = []
    def chat(self,user_text:str)->str:
        self.history.append({"role":"user","content":user_text})
        system_prompt = f"你现在完全扮演芙宁娜。你当前的心情是：【{self.npc.mood}】，好感度为{self.npc.friendship}/100。请以此状态做出符合人设的戏剧化回复。"
        messages = [{"role":"system","content":system_prompt}] + self.history
        response = ollama.chat(model = self.model_name,messages = messages)
        reply = response['message']['content']
        self.history.append({"role":"assistant","content":reply})
        if "唔" in reply or "救" in reply:
            self.npc.update_mood(-0.6)
        return reply
