class FurinaNPC:
    def __init__(self):
        self.name = "芙宁娜"
        self.dramatic_index = 80
        self.friendship = 50
        self.mood = "proud" # 当前情绪状态: proud(高傲), anxious(慌乱), gentle(温柔)
    def update_mood(self,player_input_sentiment:float):
        if player_input_sentiment < -0.5:
            self.friendship-=5
            self.mood = "anxious"
        elif player_input_sentiment > 0.5:
            self.friendship+=5
            self.mood = "gentle" if self.friendship > 75 else "proud"
 
