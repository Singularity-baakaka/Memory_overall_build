# --- START OF FILE utils.py ---
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
utils.py - 原子工具库

"""
import asyncio
import requests
import time
import json
import numpy as np
import os
import re
from typing import List, Optional, Union, Dict, Any, Tuple, Callable,AsyncGenerator
import httpx     # 请确保安装了 httpx
import aiohttp   # 请确保安装了 aiohttp
from config import get_llm_config


import aiohttp


import logging


# =====================================================================
# 1. 向量嵌入 (Embedder)
# =====================================================================

class OllamaEmbedder:
    """Ollama API 客户端 (纯工具)"""
    
    def __init__(self, model_name: str, api_url: str = "http://localhost:11434/api/embeddings", retries: int = 3):
        self.model_name = model_name
        self.api_url = api_url
        self.retries = retries

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """获取单条文本向量 (768维)"""
        if not text or not text.strip():
            return None
            
        for attempt in range(self.retries):
            try:
                resp = requests.post(
                    self.api_url,
                    json={
                        "model": self.model_name, 
                        "prompt": text,
                        "options": {"temperature": 0.0, "num_ctx": 8192} 
                    },
                    timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("embedding")
            except Exception as e:
                print(f"⚠️ [Embedder] 失败 ({attempt+1}/{self.retries}): {e}")
                if attempt < self.retries - 1:
                    time.sleep(1)
        return None

# =====================================================================
# 2. 纯数学运算 (Math)
# =====================================================================

def normalize_vector(vec: Union[List[float], np.ndarray]) -> np.ndarray:
    """归一化向量 (L2 Norm)"""
    v = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm > 0: 
        return v / norm
    return v

def cosine_similarity(v1: Union[List[float], np.ndarray], v2: Union[List[float], np.ndarray]) -> float:
    """计算余弦相似度"""
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

# =====================================================================
# 3. 存储适配工具 (Storage Helpers)
# =====================================================================

def vector_to_blob(vector: np.ndarray) -> bytes:
    """Numpy -> Bytes (SQLite Blob)"""
    return vector.astype(np.float32).tobytes()

def blob_to_vector(blob: bytes, dtype=np.float32) -> np.ndarray:
    """Bytes -> Numpy"""
    return np.frombuffer(blob, dtype=dtype)

def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

# --- ReflexDB  ---

def encode_trigger_emotions(emotions: List[str]) -> str:
    """列表 -> JSON字符串 (存入 SQLite TEXT 字段)"""
    return json.dumps(emotions, ensure_ascii=False)

def decode_trigger_emotions(emotions_json: str) -> List[str]:
    """JSON字符串 -> 列表"""
    try: 
        return json.loads(emotions_json) if emotions_json else []
    except: 
        return []
    
# --- MemoryDB ---
def split_sentences(text: str) -> List[str]:
    """
    将文本切分为句子，保留语义完整性。
    支持中英文标点：。！？!?... 等。
    """
    # 匹配句子结束符（中英文句号、问号、感叹号、省略号）
    pattern = r'[。！？!?…]+|\.{3,}'
    # 用正则分割，保留分隔符（可选，当前不保留）
    chunks = re.split(pattern, text)
    # 去除空字符串并去除首尾空格
    sentences = [s.strip() for s in chunks if s.strip()]
    return sentences

def compute_robust_embedding(narrative: str, embedder: OllamaEmbedder) -> np.ndarray:
    """通过句子切分 -> 分段嵌入 -> 加权平均，获得更稳健的语义向量"""
    sentences = split_sentences(narrative)
    if not sentences:
        return np.zeros(768, dtype=np.float32)

    vecs = []
    weights = []
    
    for i, sent in enumerate(sentences):
        vec = embedder.get_embedding(sent)
        if vec:
            # 权重规则：
            # 1. 位置权重：开头和结尾往往是核心结论
            pos_weight = 1.2 if (i == 0 or i == len(sentences) - 1) else 1.0
            # 2. 长度权重：句子越长，蕴含信息密度越高
            len_weight = min(len(sent) / 20, 1.5) 
            
            vecs.append(np.array(vec, dtype=np.float32))
            weights.append(pos_weight * len_weight)
            
    if not vecs:
        return np.zeros(768, dtype=np.float32)
        
    # 加权平均
    weighted_vec = np.average(vecs, axis=0, weights=weights)
    return normalize_vector(weighted_vec)

# =====================================================================
# 4. 文件系统原子操作 (File I/O)
# =====================================================================

def read_file_content(filepath: str, default: str = "") -> str:
    """读取文本文件，若不存在则返回默认值"""
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"⚠️ 读取文件失败 {filepath}: {e}")
        return default

def overwrite_file(filepath: str, content: str):
    """覆盖写入文件"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"⚠️ 写入文件失败 {filepath}: {e}")

def append_to_file(filepath: str, content: str):
    """追加写入文件 (自动换行)"""
    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(content + "\n")
    except Exception as e:
        print(f"⚠️ 追加文件失败 {filepath}: {e}")

def ensure_file_exists(filepath: str, initial_content: str = ""):
    """确保文件存在，不存在则创建"""
    if not os.path.exists(filepath):
        overwrite_file(filepath, initial_content)
# =====================================================================
# 5. LLM 交互原子工具 (Chat API)
# =====================================================================

def call_chat_api(
    messages: List[Dict[str, str]],
    api_url: str = None,
    api_key: str = None,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    top_p: float = None,
    frequency_penalty: float = None,
    presence_penalty: float = None,
    json_mode: bool = False,
    retries: int = 3
) -> Optional[str]:
    """
    同步调用 Chat API（用于观测器等非流式场景）
    """
    # 读取默认配置
    llm_cfg = get_llm_config()
    api_url = api_url or llm_cfg["api_url"]
    api_key = api_key or llm_cfg["api_key"]
    model = model or llm_cfg["model"]
    temperature = temperature if temperature is not None else llm_cfg.get("temperature", 1.3)
    max_tokens = max_tokens if max_tokens is not None else llm_cfg.get("max_tokens", 4000)
    top_p = top_p if top_p is not None else llm_cfg.get("top_p", 0.9)
    frequency_penalty = frequency_penalty if frequency_penalty is not None else llm_cfg.get("frequency_penalty", 0.3)
    presence_penalty = presence_penalty if presence_penalty is not None else llm_cfg.get("presence_penalty", 0.2)
    timeout = llm_cfg.get("timeout", 120)  # 从配置读取超时，默认60秒

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stream": False
    }

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(retries):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"ChatAPI 请求失败 ({attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(1)
    return None

def safe_parse_json(json_str: str, default_val: Any = None) -> Any:
    """尝试解析 JSON 字符串，失败返回默认值"""
    if not json_str: return default_val
    try:
        # 清洗可能存在的 Markdown 代码块标记
        clean_str = re.sub(r'^```json\s*', '', json_str)
        clean_str = re.sub(r'\s*```$', '', clean_str)
        return json.loads(clean_str)
    except Exception as e:
        print(f"⚠️ [JSON] 解析失败: {e}")
        return default_val
    

def extract_tag_content(text: str, tag_name: str) -> str:
    """从文本中提取 <tag>内容</tag>，极其鲁棒的正则"""
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""
def extract_dialogue_from_stream(stream_xml: str) -> str:
    """
    从完整 XML 流中提取 <user> 和 <speak> 标签内的对话内容，
    返回格式化的纯文本对话。
    """
    import re
    if not stream_xml:
        return ""
    
    # 提取 <user> 内容
    user_pattern = r'<user>(.*?)</user>'
    user_matches = re.findall(user_pattern, stream_xml, re.DOTALL)
    
    # 提取 <speak> 内容
    speak_pattern = r'<speak>(.*?)</speak>'
    speak_matches = re.findall(speak_pattern, stream_xml, re.DOTALL)
    
 
    dialogue_lines = []
    

    dialogue_lines = []
    for match in re.finditer(r'<(user|speak)>(.*?)</\1>', stream_xml, re.DOTALL):
        tag = match.group(1)
        content = match.group(2).strip()
        if tag == "user":
            dialogue_lines.append(f"用户: {content}")
        else:  # speak
            dialogue_lines.append(f"林梓墨: {content}")
    
    return "\n".join(dialogue_lines)

logger = logging.getLogger("utils")



logger = logging.getLogger("utils")

async def async_stream_chat_api_generator(
    messages: List[Dict[str, str]],
    api_url: str = None,
    api_key: str = None,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    top_p: float = None,
    frequency_penalty: float = None,
    presence_penalty: float = None
) -> AsyncGenerator[str, None]:
    """
   流式生成器，在检测到闭合标签时主动截断。
    参数可从 config 读取默认值，也可显式传入覆盖。
    """
    # 读取默认配置
    llm_cfg = get_llm_config()
    api_url = api_url or llm_cfg["api_url"]
    api_key = api_key or llm_cfg["api_key"]
    model = model or llm_cfg["model"]
    temperature = temperature if temperature is not None else llm_cfg.get("temperature", 1.3)
    max_tokens = max_tokens if max_tokens is not None else llm_cfg.get("max_tokens", 4000)
    top_p = top_p if top_p is not None else llm_cfg.get("top_p", 0.9)
    frequency_penalty = frequency_penalty if frequency_penalty is not None else llm_cfg.get("frequency_penalty", 0.3)
    presence_penalty = presence_penalty if presence_penalty is not None else llm_cfg.get("presence_penalty", 0.2)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stream": True,
    }

    # 定义要检测的闭合标签
    close_tags = ["</thought>", "</recall>", "</speak>", "<stop/>"]

    timeout = aiohttp.ClientTimeout(total=60, sock_read=60)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(api_url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"API Error: {response.status} - {error_text}")
                return

            accumulated = ""
            try:
                async for line in response.content:
                    if not line:
                        continue
                    
                    line = line.decode('utf-8').strip()
                    if not line.startswith("data: "):
                        continue
                    
                    if line == "data: [DONE]":
                        break
                    
                    try:
                        data = json.loads(line[6:])
                        delta = data['choices'][0]['delta']
                        if 'content' in delta:
                            token = delta['content']
                            accumulated += token
                            yield token
                            
                            # 检测闭合标签，遇到后主动关闭连接
                            for tag in close_tags:
                                if tag in accumulated:
                                
                                    return
                    except Exception as e:
                        logger.error(f"解析流数据异常: {e}")
                        continue
                        
            except (GeneratorExit, asyncio.CancelledError):
                # 生成器被关闭或任务取消，正常退出
                return
            except Exception as e:
                logger.error(f"流式读取异常: {e}")
                return