#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
txt_to_vectordb.py - 对话文本 → 向量记忆库 固化脚本

【整体工作流程】
  txt 对话文件
      ↓ parse_dialogue_file()       # 逐行解析，识别角色和内容
  对话列表 [{"role":..., "content":...}, ...]
      ↓ chunk_dialogues()           # 按固定轮数切片，每片成为一条记忆
  对话片段列表 [[...], [...], ...]
      ↓ generate_episode_summary()  # (LLM模式) 调用 LLM 生成摘要/氛围/关键词
      ↓ build_episode_data_fast()   # (快速模式) 直接用原文，正则提取关键词
  episode_data 字典
      ↓ MemoryDB.add_memory_episode()
  SQLite + FAISS 向量记忆库（可检索）

支持的 txt 对话格式：
    用户: 你好呀
    林梓墨: 你好～今天心情怎么样？
    用户: 挺好的，刚看完一部电影
    林梓墨: 是什么电影呀？

用法：
    python txt_to_vectordb.py  chat_log.txt  --output ./data/my_memory
    python txt_to_vectordb.py  chat_log.txt  --chunk-size 10 --fast
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from memory_db import MemoryDB
from utils import OllamaEmbedder, call_chat_api, safe_parse_json


# =====================================================================
# 1. 对话文件解析
# =====================================================================

def parse_dialogue_file(filepath: str) -> List[Dict[str, str]]:
    """
    【功能】读取 txt 对话文件，逐行解析成结构化的对话列表。

    【原理】
    用正则表达式 r'^(.+?)[：:]\s*(.*)' 匹配每一行：
      - 若行格式是 "角色名: 内容" 或 "角色名：内容"（中英文冒号均支持），
        则创建一条新对话 {"role": 角色名, "content": 内容}；
      - 若该行没有角色前缀（多行消息的续行），则直接拼接到上一条对话的
        content 末尾，用换行符连接，保持语义完整性；
      - 空行直接跳过，不影响解析。

    【效果】
    将如下文本：
        用户: 你好，今天
        天气不错
        林梓墨: 是啊！
    解析为：
        [
          {"role": "用户",   "content": "你好，今天\n天气不错"},
          {"role": "林梓墨", "content": "是啊！"}
        ]

    :param filepath: txt 文件的绝对或相对路径
    :return: 按原文顺序排列的对话字典列表，文件为空时返回 []
    :raises FileNotFoundError: 文件不存在时抛出
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"对话文件不存在: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    dialogues: List[Dict[str, str]] = []
    # 匹配 "角色名: 内容" 或 "角色名：内容"（同时支持中英文冒号）
    role_pattern = re.compile(r'^(.+?)[：:]\s*(.*)')

    for line in lines:
        line = line.rstrip('\n')
        stripped = line.strip()
        if not stripped:
            # 空行跳过，不影响后续解析
            continue

        match = role_pattern.match(stripped)
        if match:
            # 成功识别到 "角色: 内容" 格式 → 新建一条对话记录
            role = match.group(1).strip()
            content = match.group(2).strip()
            dialogues.append({"role": role, "content": content})
        else:
            # 没有角色前缀 → 这是上一条消息的续行，拼接进去
            if dialogues:
                dialogues[-1]["content"] += "\n" + stripped

    return dialogues


def chunk_dialogues(dialogues: List[Dict[str, str]],
                    chunk_size: int = 20) -> List[List[Dict[str, str]]]:
    """
    【功能】将完整对话列表按固定轮数切分成多个片段，每个片段将成为独立的一条记忆。

    【原理】
    用 Python 的切片语法 dialogues[i : i+chunk_size]，
    步长为 chunk_size，将对话列表切成若干子列表（即"记忆片段"）：
      - chunk_size=20 时：前20条为片段1，21-40条为片段2，以此类推；
      - 最后一个片段可能不足 chunk_size 条，正常保留；
      - 若 chunk_size < 2，强制设为 2（单条消息不构成有意义的对话片段）。

    【效果】
    把一段长对话拆成粒度合适的记忆单元。chunk_size 越小，记忆越细粒度但数量更多；
    chunk_size 越大，每条记忆包含更多上下文但检索精度可能下降。

    :param dialogues: parse_dialogue_file() 返回的完整对话列表
    :param chunk_size: 每个片段的最大对话条数，默认 20
    :return: 二维列表 [[片段1的对话们], [片段2的对话们], ...]
    """
    # 强制最小值：至少需要一问一答才有意义
    MIN_CHUNK_SIZE = 2
    if chunk_size < MIN_CHUNK_SIZE:
        chunk_size = MIN_CHUNK_SIZE

    chunks = []
    for i in range(0, len(dialogues), chunk_size):
        chunk = dialogues[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
    return chunks


# =====================================================================
# 2. LLM 摘要生成
# =====================================================================

# 这是发给 LLM 的提示词模板。
# 要求 LLM 扮演"记忆整理专家"，对输入的对话片段输出严格 JSON，
# 包含三个字段：narrative（叙事摘要）、atmosphere（氛围标签）、keywords（关键词列表）。
# temperature=0.3 使输出更稳定，减少随机性，避免 JSON 格式错乱。
_SUMMARY_PROMPT = """你是一个记忆整理专家。
请阅读下面的一段对话，然后用 **严格 JSON** 格式输出以下三个字段：
{
  "narrative": "用第三人称、100字以内概括这段对话的核心内容和发展",
  "atmosphere": "用2-4个情感词描述对话氛围，用逗号分隔",
  "keywords": ["关键词1", "关键词2", "关键词3"]
}

只输出 JSON，不要输出任何其他内容。

对话内容：
"""


def generate_episode_summary(dialogue_chunk: List[Dict[str, str]]) -> Optional[Dict]:
    """
    【功能】调用 LLM，为一段对话生成高质量的叙事摘要、氛围标签和关键词。

    【原理】
    1. 把对话列表拼成可读文本（每行"角色: 内容"）；
    2. 构造 messages 列表（system + user），拼入 _SUMMARY_PROMPT 提示词；
    3. 调用 call_chat_api() 发送请求，temperature=0.3 确保输出格式稳定；
    4. 用 safe_parse_json() 解析返回的 JSON 字符串
       （会自动去掉 LLM 可能包裹的 ```json ... ``` 代码块标记）；
    5. 提取并返回 narrative / atmosphere / keywords 三个字段。

    【效果】
    输入对话片段 → 输出结构化摘要，例如：
      {
        "narrative": "用户与林梓墨讨论了《时间简史》和黑洞的奥秘，氛围轻松愉快。",
        "atmosphere": "好奇, 轻松, 愉快",
        "keywords": ["时间简史", "黑洞", "霍金"]
      }
    若 LLM 调用失败或返回格式不合法，返回 None（上层会自动降级到快速模式）。

    :param dialogue_chunk: 单个对话片段（chunk_dialogues 切出的一个子列表）
    :return: 摘要字典，或 None（失败时）
    """
    # 将对话列表拼接成纯文本，方便 LLM 阅读
    text_block = "\n".join(
        f'{d["role"]}: {d["content"]}' for d in dialogue_chunk
    )

    messages = [
        {"role": "system", "content": "你是一个记忆整理专家，请严格按照用户要求输出 JSON。"},
        {"role": "user", "content": _SUMMARY_PROMPT + text_block}
    ]

    # temperature=0.3：比默认值更低，让 LLM 输出更确定，减少 JSON 格式错误概率
    raw = call_chat_api(messages, temperature=0.3, max_tokens=500)
    if not raw:
        return None  # LLM 请求失败，返回 None，上层降级到快速模式

    # safe_parse_json 会自动处理 ```json ... ``` 这类 LLM 常见的多余包裹
    parsed = safe_parse_json(raw)
    if not parsed or not isinstance(parsed, dict):
        return None  # 解析失败，同样返回 None

    return {
        "narrative":  parsed.get("narrative", ""),
        "atmosphere": parsed.get("atmosphere", ""),
        "keywords":   parsed.get("keywords", []),
    }


def build_episode_data(dialogue_chunk: List[Dict[str, str]],
                       summary: Dict,
                       timestamp: Optional[str] = None) -> Dict:
    """
    【功能】将对话片段 + LLM 摘要组装为 MemoryDB 可直接写入的 episode_data 字典。

    【原理】
    这是一个纯粹的数据组装函数（无业务逻辑），负责把两个来源的数据合并：
      - 从 dialogue_chunk 中提取原始对话列表（raw_dialogue，用于后续追溯上下文）；
      - 从 summary 中取 narrative（叙事）、atmosphere（氛围）、keywords（关键词）；
      - timestamp 若未传入则用当前时间，保证每条记忆都有时间戳；
      - importance 固定为 0.5（中等重要性），后续可按需修改。

    【效果】
    输出符合 MemoryDB.add_memory_episode() 接口规范的字典，
    可直接传入 MemoryDB 完成入库，无需再做格式转换。

    :param dialogue_chunk: 原始对话片段（保留完整的 role/content）
    :param summary: generate_episode_summary() 返回的摘要字典
    :param timestamp: 时间戳字符串，格式 "YYYY-MM-DD HH:MM:SS"，默认为当前时间
    :return: episode_data 字典，可直接传给 MemoryDB.add_memory_episode()
    """
    # 保留完整的原始对话，存入数据库供后续回溯
    raw_dialogue = [
        {"role": d["role"], "content": d["content"]}
        for d in dialogue_chunk
    ]

    return {
        "timestamp":    timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "narrative":    summary["narrative"],    # LLM 生成的第三人称叙事摘要
        "raw_dialogue": raw_dialogue,            # 原始对话备份
        "atmosphere":   summary.get("atmosphere", ""),  # 氛围标签，如 "轻松, 温馨"
        "keywords":     summary.get("keywords", []),    # 关键词列表，如 ["黑洞", "霍金"]
        "importance":   0.5,                     # 重要性权重，0~1，影响混合检索排序
    }


# =====================================================================
# 3. 直接固化（不调用 LLM 的快速模式）
# =====================================================================

def build_episode_data_fast(dialogue_chunk: List[Dict[str, str]],
                            timestamp: Optional[str] = None) -> Dict:
    """
    【功能】不依赖 LLM，直接用对话原文构建 episode_data（快速/离线模式）。

    【原理】
    这是 generate_episode_summary + build_episode_data 的轻量替代方案：
      1. narrative：直接将对话片段拼成 "角色: 内容" 多行文本，作为叙事；
      2. atmosphere：留空（无法在没有 LLM 的情况下自动推断情感）；
      3. keywords：用正则 r'[\u4e00-\u9fff]{2,6}' 从对话内容中提取
         长度 2~6 个汉字的中文词组作为候选关键词，去重后取前 5 个。
         （CJK 统一汉字范围 U+4E00–U+9FFF，覆盖绝大多数常用汉字）

    【效果】
    速度快（无网络请求），适合离线场景或 LLM 调用失败时的降级处理。
    代价是 narrative 是原文堆砌（较长），keywords 是粗粒度正则提取（可能有噪声）。

    【关键词提取示例】
    对话内容 "我在北京旅游，去了长城和故宫"
      → 正则找到: ["北京", "旅游", "长城", "故宫", ...]
      → 去重取前5个作为 keywords

    :param dialogue_chunk: 原始对话片段列表
    :param timestamp: 可选时间戳，默认为当前时间
    :return: episode_data 字典（与 build_episode_data 格式完全兼容）
    """
    # narrative = 把对话原文直接拼起来，作为"叙事"存入数据库
    narrative = "\n".join(
        f'{d["role"]}: {d["content"]}' for d in dialogue_chunk
    )

    raw_dialogue = [
        {"role": d["role"], "content": d["content"]}
        for d in dialogue_chunk
    ]

    # ---- 关键词简单提取 ----
    # 把所有对话内容合并成一个大字符串，方便统一处理
    all_content = " ".join(d["content"] for d in dialogue_chunk)
    # 正则：匹配 2~6 个连续汉字（覆盖常用词语、成语、短语）
    CJK_KEYWORD_PATTERN = r'[\u4e00-\u9fff]{2,6}'
    MAX_KEYWORDS = 5
    words = re.findall(CJK_KEYWORD_PATTERN, all_content)

    # 用 set 去重，同时保持首次出现顺序，取前 MAX_KEYWORDS 个
    seen = set()
    keywords = []
    for w in words:
        if w not in seen:
            seen.add(w)
            keywords.append(w)
        if len(keywords) >= MAX_KEYWORDS:
            break

    return {
        "timestamp":    timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "narrative":    narrative,       # 直接用原文作为叙事（不经过 LLM 提炼）
        "raw_dialogue": raw_dialogue,    # 原始对话备份
        "atmosphere":   "",              # 快速模式无法推断氛围，留空
        "keywords":     keywords,        # 正则提取的中文关键词
        "importance":   0.5,
    }


# =====================================================================
# 4. 主流程
# =====================================================================

def txt_to_vectordb(txt_path: str,
                    output_dir: str = "./data/memory_db",
                    chunk_size: int = 20,
                    use_llm: bool = True,
                    embedder: Optional[OllamaEmbedder] = None) -> Dict:
    """
    【功能】一键将 txt 对话文件固化为可检索的向量记忆库（整个项目的主入口函数）。

    【原理 / 完整流程】
      Step 1. parse_dialogue_file()  → 解析 txt，得到对话列表
      Step 2. chunk_dialogues()      → 按 chunk_size 切片
      Step 3. 对每个片段：
              - use_llm=True  → generate_episode_summary() 调 LLM 生成摘要
                                若 LLM 失败 → 自动降级到快速模式
              - use_llm=False → build_episode_data_fast() 直接构建
      Step 4. db.add_memory_episode() → 写入 SQLite + 三个 FAISS 索引
      Step 5. 统计并返回结果字典

    【效果】
    执行完毕后，output_dir 目录中会生成：
      - memory_episodes.db   （SQLite，存储结构化记忆数据）
      - semantic.faiss       （叙事语义向量索引）
      - atmosphere.faiss     （氛围向量索引）
      - keyword.faiss        （关键词向量索引）
      - *_map.json           （FAISS ID → SQLite ID 映射表）
    之后即可用 MemoryDB 对记忆库进行语义/氛围/关键词/混合检索。

    :param txt_path:   输入的 txt 对话文件路径
    :param output_dir: 向量库的输出目录，不存在会自动创建
    :param chunk_size: 每条记忆最多包含几条对话，越小粒度越细
    :param use_llm:    True=调用 LLM 生成高质量摘要；False=快速模式不调 LLM
    :param embedder:   自定义嵌入器（不传则用默认的 OllamaEmbedder）
    :return: 统计信息字典 {status, total_dialogues, total_chunks, success, failed, db_stats}
    """
    print(f"📖 正在解析对话文件: {txt_path}")
    dialogues = parse_dialogue_file(txt_path)
    if not dialogues:
        print("⚠️ 对话文件为空或格式不正确")
        return {"status": "error", "message": "empty dialogue"}

    print(f"   解析到 {len(dialogues)} 条对话")

    # Step 2: 切片 —— 把长对话分成若干记忆片段
    chunks = chunk_dialogues(dialogues, chunk_size)
    print(f"   切分为 {len(chunks)} 个记忆片段（每段最多 {chunk_size} 条对话）")

    # Step 3 & 4: 初始化数据库，逐片处理并写入
    db = MemoryDB(db_path=output_dir, embedder=embedder)
    success_count = 0
    fail_count = 0

    for idx, chunk in enumerate(chunks):
        print(f"\n🔄 处理片段 {idx + 1}/{len(chunks)} ({len(chunk)} 条对话)...")

        if use_llm:
            # LLM 模式：让模型提炼摘要，质量更高
            summary = generate_episode_summary(chunk)
            if summary:
                episode = build_episode_data(chunk, summary)
            else:
                # LLM 调用失败时的容错降级：改用快速模式，保证流程不中断
                print(f"   ⚠️ LLM 摘要失败，使用快速模式替代")
                episode = build_episode_data_fast(chunk)
        else:
            # 快速模式：直接用原文，无需网络，适合离线或批量处理
            episode = build_episode_data_fast(chunk)

        try:
            db_id = db.add_memory_episode(episode)
            print(f"   ✅ 写入成功 → DB ID: {db_id}")
            success_count += 1
        except Exception as e:
            print(f"   ❌ 写入失败: {e}")
            fail_count += 1

    # Step 5: 汇总统计信息
    stats = db.stats()
    db.close()

    result = {
        "status":          "done",
        "input_file":      txt_path,
        "total_dialogues": len(dialogues),
        "total_chunks":    len(chunks),
        "success":         success_count,
        "failed":          fail_count,
        "db_stats":        stats,
    }

    print(f"\n{'='*50}")
    print(f"✅ 固化完成!")
    print(f"   对话总数: {len(dialogues)}")
    print(f"   记忆片段: {len(chunks)}")
    print(f"   成功写入: {success_count}")
    print(f"   失败: {fail_count}")
    print(f"   存储路径: {output_dir}")
    print(f"   向量库统计: {stats}")

    return result


# =====================================================================
# 5. CLI 入口
# =====================================================================

def main():
    """
    【功能】命令行入口，解析 argparse 参数后调用 txt_to_vectordb()。

    【用法示例】
      # LLM 模式（默认），chunk_size=20，输出到 ./data/memory_db
      python txt_to_vectordb.py chat_log.txt

      # 指定输出目录和片段大小
      python txt_to_vectordb.py chat_log.txt --output ./my_db --chunk-size 10

      # 快速模式（不调 LLM，适合离线）
      python txt_to_vectordb.py chat_log.txt --fast

    【参数说明】
      txt_path        必填，txt 对话文件路径
      --output / -o   向量库输出目录，默认 ./data/memory_db
      --chunk-size / -c  每条记忆最多包含的对话轮数，默认 20
      --fast          加此 flag 则不调用 LLM（use_llm=False）
    """
    parser = argparse.ArgumentParser(
        description="将 txt 对话文件固化为向量记忆库"
    )
    parser.add_argument("txt_path", help="txt 对话文件路径")
    parser.add_argument(
        "--output", "-o",
        default="./data/memory_db",
        help="向量库输出目录 (默认: ./data/memory_db)"
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int, default=20,
        help="每条记忆包含的最大对话轮数 (默认: 20)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="快速模式：不调用 LLM，直接用原文入库"
    )

    args = parser.parse_args()

    txt_to_vectordb(
        txt_path=args.txt_path,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        use_llm=not args.fast,  # --fast flag 存在时 use_llm=False
    )


if __name__ == "__main__":
    main()
