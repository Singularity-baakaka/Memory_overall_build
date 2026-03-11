#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
txt_to_vectordb.py - 对话文本 → 向量记忆库 固化脚本

功能：读取 txt 格式的对话文件，自动解析对话轮次，调用 LLM 生成
叙事摘要 / 氛围标签 / 关键词，然后写入 MemoryDB（SQLite + FAISS）
实现"一键把聊天记录变成可检索的向量记忆库"。

支持的 txt 对话格式：
    用户: 你好呀
    林梓墨: 你好～今天心情怎么样？
    用户: 挺好的，刚看完一部电影
    林梓墨: 是什么电影呀？

用法：
    python txt_to_vectordb.py  chat_log.txt  --output ./data/my_memory
    python txt_to_vectordb.py  chat_log.txt  --chunk-size 10
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
    解析 txt 对话文件，返回按顺序排列的对话列表。

    支持格式：
      - "角色名: 内容"  （中文冒号或英文冒号）
      - "角色名：内容"
    多行内容（无角色前缀的行）会被拼接到上一条对话。

    :param filepath: txt 文件路径
    :return: [{"role": "用户", "content": "..."}, {"role": "林梓墨", "content": "..."}, ...]
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"对话文件不存在: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    dialogues: List[Dict[str, str]] = []
    # 匹配 "角色名: 内容" 或 "角色名：内容"
    role_pattern = re.compile(r'^(.+?)[：:]\s*(.*)')

    for line in lines:
        line = line.rstrip('\n')
        stripped = line.strip()
        if not stripped:
            continue

        match = role_pattern.match(stripped)
        if match:
            role = match.group(1).strip()
            content = match.group(2).strip()
            dialogues.append({"role": role, "content": content})
        else:
            # 没有角色前缀的行 → 拼接到上一条消息
            if dialogues:
                dialogues[-1]["content"] += "\n" + stripped

    return dialogues


def chunk_dialogues(dialogues: List[Dict[str, str]],
                    chunk_size: int = 20) -> List[List[Dict[str, str]]]:
    """
    将对话列表按固定轮次切分为多个片段（chunk），每个片段作为一条记忆。

    :param dialogues: 完整的对话列表
    :param chunk_size: 每个片段包含的最大对话条数
    :return: [[{role, content}, ...], ...]  二维列表
    """
    if chunk_size < 2:
        chunk_size = 2
    chunks = []
    for i in range(0, len(dialogues), chunk_size):
        chunk = dialogues[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
    return chunks


# =====================================================================
# 2. LLM 摘要生成
# =====================================================================

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
    调用 LLM 对一段对话生成叙事摘要、氛围标签和关键词。

    :param dialogue_chunk: 对话列表 [{"role": ..., "content": ...}, ...]
    :return: {"narrative": str, "atmosphere": str, "keywords": list} 或 None
    """
    # 拼成可读文本
    text_block = "\n".join(
        f'{d["role"]}: {d["content"]}' for d in dialogue_chunk
    )

    messages = [
        {"role": "system", "content": "你是一个记忆整理专家，请严格按照用户要求输出 JSON。"},
        {"role": "user", "content": _SUMMARY_PROMPT + text_block}
    ]

    raw = call_chat_api(messages, temperature=0.3, max_tokens=500)
    if not raw:
        return None

    parsed = safe_parse_json(raw)
    if not parsed or not isinstance(parsed, dict):
        return None

    return {
        "narrative": parsed.get("narrative", ""),
        "atmosphere": parsed.get("atmosphere", ""),
        "keywords": parsed.get("keywords", []),
    }


def build_episode_data(dialogue_chunk: List[Dict[str, str]],
                       summary: Dict,
                       timestamp: Optional[str] = None) -> Dict:
    """
    将对话片段和 LLM 摘要组装为 MemoryDB.add_memory_episode 所需的字典。

    :param dialogue_chunk: 原始对话片段
    :param summary: LLM 生成的摘要字典
    :param timestamp: 可选时间戳，默认使用当前时间
    :return: episode_data 字典
    """
    raw_dialogue = [
        {"role": d["role"], "content": d["content"]}
        for d in dialogue_chunk
    ]

    return {
        "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "narrative": summary["narrative"],
        "raw_dialogue": raw_dialogue,
        "atmosphere": summary.get("atmosphere", ""),
        "keywords": summary.get("keywords", []),
        "importance": 0.5,
    }


# =====================================================================
# 3. 直接固化（不调用 LLM 的快速模式）
# =====================================================================

def build_episode_data_fast(dialogue_chunk: List[Dict[str, str]],
                            timestamp: Optional[str] = None) -> Dict:
    """
    不调用 LLM，直接用对话原文作为 narrative 快速构建 episode_data。
    适用于无法连接 LLM 或希望快速入库的场景。

    :param dialogue_chunk: 原始对话片段
    :param timestamp: 可选时间戳
    :return: episode_data 字典
    """
    narrative = "\n".join(
        f'{d["role"]}: {d["content"]}' for d in dialogue_chunk
    )

    raw_dialogue = [
        {"role": d["role"], "content": d["content"]}
        for d in dialogue_chunk
    ]

    # 简单提取关键词：取所有内容中出现的名词性短语（简化版：取前几个角色名和长短语）
    all_content = " ".join(d["content"] for d in dialogue_chunk)
    # 提取较长的词组作为关键词
    words = re.findall(r'[\u4e00-\u9fff]{2,6}', all_content)
    # 去重并取前 5 个
    seen = set()
    keywords = []
    for w in words:
        if w not in seen:
            seen.add(w)
            keywords.append(w)
        if len(keywords) >= 5:
            break

    return {
        "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "narrative": narrative,
        "raw_dialogue": raw_dialogue,
        "atmosphere": "",
        "keywords": keywords,
        "importance": 0.5,
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
    将 txt 对话文件固化为向量记忆库（一键入库）。

    :param txt_path: txt 对话文件路径
    :param output_dir: 向量库输出目录
    :param chunk_size: 每条记忆包含的最大对话轮数
    :param use_llm: 是否调用 LLM 生成摘要（False 则用快速模式）
    :param embedder: 可选的嵌入器实例
    :return: 统计信息字典
    """
    print(f"📖 正在解析对话文件: {txt_path}")
    dialogues = parse_dialogue_file(txt_path)
    if not dialogues:
        print("⚠️ 对话文件为空或格式不正确")
        return {"status": "error", "message": "empty dialogue"}

    print(f"   解析到 {len(dialogues)} 条对话")

    # 切分
    chunks = chunk_dialogues(dialogues, chunk_size)
    print(f"   切分为 {len(chunks)} 个记忆片段（每段最多 {chunk_size} 条对话）")

    # 初始化 MemoryDB
    db = MemoryDB(db_path=output_dir, embedder=embedder)
    success_count = 0
    fail_count = 0

    for idx, chunk in enumerate(chunks):
        print(f"\n🔄 处理片段 {idx + 1}/{len(chunks)} ({len(chunk)} 条对话)...")

        if use_llm:
            summary = generate_episode_summary(chunk)
            if summary:
                episode = build_episode_data(chunk, summary)
            else:
                print(f"   ⚠️ LLM 摘要失败，使用快速模式替代")
                episode = build_episode_data_fast(chunk)
        else:
            episode = build_episode_data_fast(chunk)

        try:
            db_id = db.add_memory_episode(episode)
            print(f"   ✅ 写入成功 → DB ID: {db_id}")
            success_count += 1
        except Exception as e:
            print(f"   ❌ 写入失败: {e}")
            fail_count += 1

    # 统计
    stats = db.stats()
    db.close()

    result = {
        "status": "done",
        "input_file": txt_path,
        "total_dialogues": len(dialogues),
        "total_chunks": len(chunks),
        "success": success_count,
        "failed": fail_count,
        "db_stats": stats,
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
        use_llm=not args.fast,
    )


if __name__ == "__main__":
    main()
