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
    """
    Ollama API 客户端 (纯工具)
    
    原理：通过 HTTP 请求调用本地部署的 Ollama 嵌入模型服务，
    将任意文本转换为固定维度（768维）的浮点向量表示。
    该向量可用于后续的语义相似度计算和向量检索。
    """
    
    def __init__(self, model_name: str, api_url: str = "http://localhost:11434/api/embeddings", retries: int = 3):
        """
        初始化 Ollama 嵌入器实例。

        原理：记录模型名称、API 端点地址和最大重试次数，
        后续调用 get_embedding 时将使用这些配置连接本地 Ollama 服务。

        :param model_name: Ollama 中已部署的嵌入模型名称（如 'gte-base-zh'）
        :param api_url: Ollama 嵌入 API 的地址，默认为本地 11434 端口
        :param retries: 请求失败时的最大重试次数，默认 3 次
        """
        self.model_name = model_name
        self.api_url = api_url
        self.retries = retries

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        获取单条文本的向量嵌入（768维浮点数列表）。

        原理：将输入文本发送到 Ollama Embedding API，模型会将文本编码为
        一个 768 维的稠密向量。该向量捕获了文本的语义信息，语义相近的文本
        会被映射到向量空间中距离较近的位置。
        使用 temperature=0.0 确保嵌入结果的确定性（同一文本始终返回相同向量）。
        内置重试机制：若请求失败，会间隔 1 秒后重试，最多重试 self.retries 次。

        :param text: 待嵌入的文本
        :return: 768维浮点数列表，若失败或文本为空则返回 None
        """
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
    """
    对向量进行 L2 归一化，使其模长变为 1。

    原理：L2 归一化是将向量除以其欧几里得范数（L2 Norm）。
    归一化后的向量保留了方向信息但模长为 1，这样两个归一化向量
    的点积就等于它们的余弦相似度，极大简化了后续的相似度计算。
    FAISS 的 IndexFlatIP（内积索引）在输入归一化向量时，
    其内积检索结果等价于余弦相似度检索。
    若向量范数为 0（零向量），则直接返回原向量以避免除零错误。

    :param vec: 输入向量，可以是列表或 numpy 数组
    :return: 归一化后的 numpy float32 数组
    """
    v = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm > 0: 
        return v / norm
    return v

def cosine_similarity(v1: Union[List[float], np.ndarray], v2: Union[List[float], np.ndarray]) -> float:
    """
    计算两个向量之间的余弦相似度。

    原理：余弦相似度 = (v1 · v2) / (||v1|| * ||v2||)，衡量两个向量在方向上的一致性。
    值域为 [-1, 1]：1 表示方向完全相同（语义最相似），0 表示正交（无关），
    -1 表示方向完全相反。分母添加极小值 1e-8 以防止除零错误。
    广泛用于文本语义相似度比较，因为它不受向量长度影响，只关注方向。

    :param v1: 第一个向量
    :param v2: 第二个向量
    :return: 余弦相似度值（浮点数）
    """
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

# =====================================================================
# 3. 存储适配工具 (Storage Helpers)
# =====================================================================

def vector_to_blob(vector: np.ndarray) -> bytes:
    """
    将 Numpy 向量转换为二进制字节串，用于存入 SQLite 的 BLOB 字段。

    原理：Numpy 数组的 tobytes() 方法将数组按内存布局序列化为连续的
    二进制数据。先将数据类型统一为 float32（每个元素 4 字节），确保
    存储格式一致，便于后续用 blob_to_vector 反序列化。
    768 维 float32 向量将被转换为 3072 字节的二进制数据。

    :param vector: Numpy 向量数组
    :return: 二进制字节串
    """
    return vector.astype(np.float32).tobytes()

def blob_to_vector(blob: bytes, dtype=np.float32) -> np.ndarray:
    """
    将二进制字节串还原为 Numpy 向量，是 vector_to_blob 的逆操作。

    原理：使用 np.frombuffer 将原始字节数据按指定的数据类型（默认 float32）
    解释为 Numpy 数组。要求传入的 blob 必须是由 vector_to_blob 生成的，
    以确保字节长度和数据类型匹配。

    :param blob: 二进制字节串（来自 SQLite BLOB 字段）
    :param dtype: 数据类型，默认 np.float32
    :return: Numpy 向量数组
    """
    return np.frombuffer(blob, dtype=dtype)

def ensure_dir(path: str):
    """
    确保指定的目录路径存在，若不存在则递归创建。

    原理：调用 os.makedirs 并设置 exist_ok=True，这样即使目录已存在
    也不会抛出异常。用于在写入文件前确保目标目录的存在性，避免 FileNotFoundError。

    :param path: 需要确保存在的目录路径
    """
    os.makedirs(path, exist_ok=True)

# --- ReflexDB  ---

def encode_trigger_emotions(emotions: List[str]) -> str:
    """
    将情绪标签列表编码为 JSON 字符串，用于存入 SQLite 的 TEXT 字段。

    原理：SQLite 没有原生的数组类型，因此将 Python 列表通过 json.dumps
    序列化为 JSON 字符串进行存储。设置 ensure_ascii=False 以正确保留
    中文等非 ASCII 字符，避免被转义为 Unicode 编码（如 \\u4f60）。

    :param emotions: 情绪标签列表，如 ['开心', '惊讶']
    :return: JSON 格式字符串，如 '["开心", "惊讶"]'
    """
    return json.dumps(emotions, ensure_ascii=False)

def decode_trigger_emotions(emotions_json: str) -> List[str]:
    """
    将 JSON 字符串解码回情绪标签列表，是 encode_trigger_emotions 的逆操作。

    原理：使用 json.loads 将 JSON 字符串反序列化为 Python 列表。
    内置容错处理：若输入为空或解析失败（如数据损坏），返回空列表而非抛出异常，
    确保上层逻辑不会因数据格式问题而中断。

    :param emotions_json: JSON 格式的情绪字符串
    :return: 情绪标签列表，解析失败时返回空列表
    """
    try: 
        return json.loads(emotions_json) if emotions_json else []
    except: 
        return []
    
# --- MemoryDB ---
def split_sentences(text: str) -> List[str]:
    """
    将文本按句子边界切分为句子列表，保留语义完整性。

    原理：使用正则表达式匹配中英文常见的句子结束符（句号、问号、感叹号、省略号），
    以这些标点为分隔符对文本进行切分。切分后去除每个片段的首尾空白，
    并过滤掉空字符串。此函数用于 compute_robust_embedding 中，将长文本
    拆分为语义独立的句子单元，以便逐句嵌入后进行加权平均。

    :param text: 待切分的文本
    :return: 句子列表（不含分隔符）
    """
    # 匹配句子结束符（中英文句号、问号、感叹号、省略号）
    pattern = r'[。！？!?…]+|\.{3,}'
    # 用正则分割，保留分隔符（可选，当前不保留）
    chunks = re.split(pattern, text)
    # 去除空字符串并去除首尾空格
    sentences = [s.strip() for s in chunks if s.strip()]
    return sentences

def compute_robust_embedding(narrative: str, embedder: OllamaEmbedder) -> np.ndarray:
    """
    通过句子切分 → 分段嵌入 → 加权平均，获得更稳健的语义向量。

    原理：直接对整段长文本做嵌入，模型可能丢失局部细节或被无关内容稀释。
    本函数采用"分而治之"策略：
    1. 先用 split_sentences 将文本切分为独立句子；
    2. 对每个句子单独调用嵌入模型获取向量；
    3. 为每个句子分配组合权重（位置权重 × 长度权重）：
       - 位置权重：首句和尾句权重为 1.2（通常是核心观点/结论），中间句为 1.0；
       - 长度权重：句子越长信息密度越高，权重 = min(字符数/20, 1.5)；
    4. 对所有句子向量做加权平均，再 L2 归一化得到最终向量。
    这种方法比直接嵌入整段文本能更好地捕获核心语义。

    :param narrative: 待嵌入的叙事文本
    :param embedder: OllamaEmbedder 实例
    :return: 归一化后的 768 维语义向量，若无有效句子则返回零向量
    """
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
    """
    安全地读取文本文件内容。

    原理：先检查文件是否存在，存在则以 UTF-8 编码读取并去除首尾空白。
    若文件不存在或读取过程中出现异常（如权限不足、编码错误），
    返回预设的默认值而非抛出异常，确保上层调用不会因文件问题中断。

    :param filepath: 文件路径
    :param default: 文件不存在或读取失败时的默认返回值
    :return: 文件文本内容或默认值
    """
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"⚠️ 读取文件失败 {filepath}: {e}")
        return default

def overwrite_file(filepath: str, content: str):
    """
    以覆盖模式写入文件，原有内容将被完全替换。

    原理：使用 'w' 模式打开文件（写入模式），会清空文件原有内容后写入新内容。
    若文件不存在则自动创建。采用 UTF-8 编码确保中文等字符正确存储。
    包含异常捕获，写入失败时打印警告信息而非中断程序。

    :param filepath: 文件路径
    :param content: 要写入的文本内容
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"⚠️ 写入文件失败 {filepath}: {e}")

def append_to_file(filepath: str, content: str):
    """
    以追加模式写入文件，新内容添加到文件末尾并自动换行。

    原理：使用 'a' 模式打开文件（追加模式），在不影响原有内容的前提下，
    将新内容追加到文件末尾，并自动添加换行符。适用于日志记录、对话历史
    等需要持续累积内容的场景。若文件不存在则自动创建。

    :param filepath: 文件路径
    :param content: 要追加的文本内容
    """
    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(content + "\n")
    except Exception as e:
        print(f"⚠️ 追加文件失败 {filepath}: {e}")

def ensure_file_exists(filepath: str, initial_content: str = ""):
    """
    确保文件存在，若不存在则用初始内容创建。

    原理：检查文件是否存在，若不存在则调用 overwrite_file 用给定的初始内容
    创建文件。这是一个幂等操作——多次调用不会影响已存在的文件内容，
    仅在文件缺失时进行创建。适用于首次启动时初始化配置文件或数据文件。

    :param filepath: 文件路径
    :param initial_content: 文件不存在时写入的初始内容，默认为空字符串
    """
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
    同步调用 Chat API，获取 LLM 的完整回复（非流式）。

    原理：向兼容 OpenAI 格式的 Chat Completions API 发送 HTTP POST 请求，
    传入对话消息列表（messages），由 LLM 生成回复。
    工作流程：
    1. 从 config.py 读取默认 LLM 配置（API 地址、密钥、模型名等），
       显式传入的参数会覆盖默认配置；
    2. 构造请求 payload，包含模型参数（temperature 控制随机性、top_p 核采样、
       frequency/presence_penalty 控制重复度等）；
    3. 若 json_mode=True，添加 response_format 指示模型以 JSON 格式输出；
    4. 发送请求并提取回复内容（choices[0].message.content）；
    5. 内置重试机制：请求失败时间隔 1 秒后重试，最多 retries 次。

    :param messages: 对话消息列表，格式为 [{"role": "system/user/assistant", "content": "..."}]
    :param api_url: API 端点地址（可选，默认从 config 读取）
    :param api_key: API 密钥（可选，默认从 config 读取）
    :param model: 模型名称（可选，默认从 config 读取）
    :param temperature: 温度参数，控制输出随机性，越高越随机
    :param max_tokens: 最大生成 token 数
    :param top_p: 核采样参数
    :param frequency_penalty: 频率惩罚，降低重复内容
    :param presence_penalty: 存在惩罚，鼓励新话题
    :param json_mode: 是否要求模型以 JSON 格式输出
    :param retries: 最大重试次数
    :return: LLM 的回复文本，失败则返回 None
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
    """
    鲁棒地解析 JSON 字符串，自动清洗 LLM 常见的格式污染。

    原理：LLM（大语言模型）返回的 JSON 内容经常被包裹在 Markdown 代码块标记中
    （如 ```json ... ```），直接用 json.loads 解析会失败。
    本函数先用正则表达式去除开头的 ```json 和结尾的 ``` 标记，
    再进行 JSON 解析。若仍然失败（如 JSON 格式本身损坏），
    返回默认值而非抛出异常，保证上层逻辑的稳定性。

    :param json_str: 待解析的 JSON 字符串（可能带有 Markdown 代码块标记）
    :param default_val: 解析失败时的默认返回值
    :return: 解析后的 Python 对象（dict/list/str 等），失败返回 default_val
    """
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
    """
    从文本中提取指定 XML 标签内的内容。

    原理：使用正则表达式匹配 <tag_name>...</tag_name> 格式的标签对，
    提取标签内部的文本内容。re.DOTALL 标志使 '.' 能匹配换行符，
    支持跨行的标签内容。'.*?' 非贪婪匹配确保在有多个同名标签时
    匹配到最短的内容段。主要用于解析 LLM 输出中的结构化标签
    （如 <thought>、<speak>、<recall> 等）。

    :param text: 包含 XML 标签的文本
    :param tag_name: 要提取的标签名（不含尖括号）
    :return: 标签内的文本内容（已去除首尾空白），未匹配到则返回空字符串
    """
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_dialogue_from_stream(stream_xml: str) -> str:
    """
    从完整 XML 流中提取 <user> 和 <speak> 标签内的对话内容，
    返回格式化的纯文本对话。

    原理：LLM 的流式输出通常包含多种 XML 标签（如 <thought>、<user>、<speak>），
    本函数使用正则表达式 re.finditer 按出现顺序遍历所有 <user> 和 <speak> 标签，
    提取其中的文本内容，并分别添加 "用户:" 或 "林梓墨:" 前缀，
    最终按原始出现顺序拼接为可读的对话文本。
    这样可以从混杂着思考过程等内部标签的原始输出中，
    提取出纯净的对话记录用于存档或展示。

    :param stream_xml: 包含 XML 标签的完整流式输出文本
    :return: 格式化的对话文本，每行一条发言，空输入返回空字符串
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
    异步流式调用 Chat API 的生成器，逐 token 输出 LLM 回复，
    并在检测到闭合标签时主动截断。

    原理：与 call_chat_api 不同，本函数使用 SSE（Server-Sent Events）流式协议
    接收 LLM 的逐 token 输出，实现"打字机效果"的实时响应。
    工作流程：
    1. 从 config.py 读取默认配置，显式参数可覆盖默认值；
    2. 设置 stream=True 启用流式输出模式；
    3. 通过 aiohttp 异步发送请求，逐行读取 SSE 数据流；
    4. 每行 SSE 数据格式为 "data: {json}"，解析出 delta.content 中的 token；
    5. 用 yield 逐个返回 token 给调用者，实现流式输出；
    6. 同时维护一个 accumulated 缓冲区，累积已接收的所有 token；
    7. 每收到新 token 后检查缓冲区是否包含预定义的闭合标签
       （</thought>、</recall>、</speak>、<stop/>），
       一旦检测到则主动 return 终止生成器，切断与 API 的连接。
       这种"主动截断"机制可以节省 token 消耗，防止模型继续生成冗余内容。
    8. 内置异常处理，优雅应对连接超时、生成器取消等情况。

    :param messages: 对话消息列表
    :param api_url: API 端点地址（可选）
    :param api_key: API 密钥（可选）
    :param model: 模型名称（可选）
    :param temperature: 温度参数（可选）
    :param max_tokens: 最大生成 token 数（可选）
    :param top_p: 核采样参数（可选）
    :param frequency_penalty: 频率惩罚（可选）
    :param presence_penalty: 存在惩罚（可选）
    :yields: LLM 逐个生成的 token 字符串
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