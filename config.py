import os
# ==================== LLM API 配置 (DeepSeek) ====================
LLM_CONFIG = {
    "api_key": os.getenv("DEEPSEEK_API_KEY"),             # 把你的 API Key 填在这里
    "api_url": "https://api.deepseek.com/v1/chat/completions",
    "model": "deepseek-chat",
    "temperature": 1.3,
    "max_tokens": 4000,
    "top_p": 0.9,              # 核采样，与 temperature 配合使用
    "frequency_penalty": 0.3,  # 频率惩罚，减少重复
    "presence_penalty": 0.2,   # 存在惩罚，鼓励新话题
    "timeout": 120             # 请求超时时间 (秒)
}

def get_llm_config():
    return LLM_CONFIG