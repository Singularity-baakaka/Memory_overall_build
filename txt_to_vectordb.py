#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
txt_to_vectordb.py - 对话文本固化为向量记忆库

功能：读取 txt 格式的对话文件，自动解析对话轮次，
按可配置的分段策略切分为记忆片段（episode），
逐段写入 MemoryDB（SQLite + FAISS），实现对话内容的向量化固化。

支持的对话格式：
  - "用户: 内容" / "角色名: 内容"（中文冒号或英文冒号）
  - "User: content" / "Assistant: content"
  - 多行续写（无说话人前缀的行自动归入上一条发言）

用法示例：
  python txt_to_vectordb.py dialogue.txt --db_path ./data/my_memory
  python txt_to_vectordb.py dialogue.txt --turns_per_segment 5
"""

import os
import re
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

from memory_db import MemoryDB
from utils import OllamaEmbedder

# 说话人标签正则：匹配 "角色名: " 或 "角色名：" 开头的行
_SPEAKER_RE = re.compile(r'^([\w\u4e00-\u9fff]+)\s*[：:]\s*(.*)$')


def parse_dialogue_file(filepath: str) -> List[Dict[str, str]]:
    """
    解析 txt 对话文件，返回对话轮次列表。

    每个轮次为 {"role": "说话人", "content": "发言内容"} 字典。
    自动识别说话人前缀（支持中英文冒号），无前缀的行作为
    上一条发言的续写内容拼接。

    :param filepath: txt 对话文件路径
    :return: 对话轮次列表
    :raises FileNotFoundError: 文件不存在
    :raises ValueError: 文件内容为空或无法解析出任何对话
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"对话文件不存在: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    turns: List[Dict[str, str]] = []
    current_role = None
    current_content_lines: List[str] = []

    for raw_line in lines:
        line = raw_line.rstrip('\n\r')

        # 跳过纯空行（但先保存之前的轮次）
        if not line.strip():
            if current_role and current_content_lines:
                turns.append({
                    "role": current_role,
                    "content": "\n".join(current_content_lines).strip()
                })
                current_role = None
                current_content_lines = []
            continue

        match = _SPEAKER_RE.match(line)
        if match:
            # 保存上一条发言
            if current_role and current_content_lines:
                turns.append({
                    "role": current_role,
                    "content": "\n".join(current_content_lines).strip()
                })
            current_role = match.group(1)
            current_content_lines = [match.group(2)] if match.group(2) else []
        else:
            # 续写行：归入当前发言
            if current_role is not None:
                current_content_lines.append(line)
            else:
                # 文件起始无角色标记时，标记为 "未知"
                current_role = "未知"
                current_content_lines.append(line)

    # 尾部残留
    if current_role and current_content_lines:
        turns.append({
            "role": current_role,
            "content": "\n".join(current_content_lines).strip()
        })

    if not turns:
        raise ValueError(f"未能从文件中解析出任何对话轮次: {filepath}")

    return turns


def segment_dialogue(turns: List[Dict[str, str]],
                     turns_per_segment: int = 10) -> List[List[Dict[str, str]]]:
    """
    将对话轮次列表按固定轮数分段。

    每段包含 turns_per_segment 个轮次（最后一段可能不足）。
    分段的目的是将长对话切分为语义粒度适中的记忆片段，
    避免单条记忆过长导致向量化时语义稀释。

    :param turns: 对话轮次列表
    :param turns_per_segment: 每段包含的轮次数量，默认 10
    :return: 分段后的二维列表
    """
    if turns_per_segment < 1:
        turns_per_segment = 1

    segments: List[List[Dict[str, str]]] = []
    for i in range(0, len(turns), turns_per_segment):
        segment = turns[i:i + turns_per_segment]
        if segment:
            segments.append(segment)
    return segments


def extract_simple_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """
    从文本中提取简易关键词（基于高频名词/短语的启发式方法）。

    原理：对文本按标点切句后，提取长度 >=2 的连续中文子串，
    按出现频率排序取 top-N。这是一种无需外部分词模型的轻量实现，
    适用于不依赖 LLM 的离线场景。

    :param text: 输入文本
    :param max_keywords: 最大关键词数量
    :return: 关键词列表
    """
    # 检测是否包含中文
    zh_segments = re.findall(r'[\u4e00-\u9fff]+', text)
    if zh_segments:
        # 对中文内容，从每个连续中文片段中提取 2~4 字的 n-gram
        ngrams: List[str] = []
        for seg in zh_segments:
            for n in (2, 3, 4):
                for i in range(len(seg) - n + 1):
                    ngrams.append(seg[i:i + n])
        candidates = ngrams
    else:
        # 对英文内容，提取单词（长度 >= 3）
        words = re.findall(r'[a-zA-Z]{3,}', text.lower())
        # 去除常见停用词
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you',
                      'all', 'can', 'her', 'was', 'one', 'our', 'out',
                      'has', 'have', 'had', 'this', 'that', 'with', 'from'}
        candidates = [w for w in words if w not in stop_words]

    # 统计频率
    freq: Dict[str, int] = {}
    for item in candidates:
        freq[item] = freq.get(item, 0) + 1

    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in sorted_items[:max_keywords]]


def build_episode_from_segment(segment: List[Dict[str, str]],
                               segment_index: int,
                               source_file: str,
                               base_time: Optional[datetime] = None) -> dict:
    """
    将一个对话分段构建为 MemoryDB 所需的 episode_data 字典。

    :param segment: 对话轮次列表（单个分段）
    :param segment_index: 分段序号（从 0 开始）
    :param source_file: 来源文件名（用于元数据标记）
    :param base_time: 基准时间，若不传则使用当前时间
    :return: episode_data 字典，可直接传入 MemoryDB.add_memory_episode()
    """
    if base_time is None:
        base_time = datetime.now()

    # 构建叙事文本（narrative）：拼接所有轮次为可读文本
    narrative_lines = []
    for turn in segment:
        narrative_lines.append(f"{turn['role']}: {turn['content']}")
    narrative = "\n".join(narrative_lines)

    # 构建原始对话备份
    raw_dialogue = [{"role": t["role"], "content": t["content"]} for t in segment]

    # 提取简易关键词
    full_text = " ".join(t["content"] for t in segment)
    keywords = extract_simple_keywords(full_text)

    # 时间戳：基准时间 + 分段序号偏移
    timestamp = (base_time + timedelta(minutes=segment_index)).isoformat()

    return {
        "timestamp": timestamp,
        "narrative": narrative,
        "raw_dialogue": raw_dialogue,
        "atmosphere": "",
        "keywords": keywords,
        "importance": 0.5,
    }


def txt_to_vectordb(
    txt_path: str,
    db_path: str = "./data/memory_db",
    turns_per_segment: int = 10,
    embedder: Optional[OllamaEmbedder] = None,
) -> dict:
    """
    核心入口：将 txt 对话文件解析并固化为向量记忆库。

    工作流程：
    1. 读取并解析对话文件 → 得到对话轮次列表
    2. 按 turns_per_segment 分段
    3. 为每段构建 episode_data
    4. 调用 MemoryDB.add_memory_episode 逐段写入
    5. 返回统计摘要

    :param txt_path: txt 对话文件路径
    :param db_path: 向量库输出目录，默认 ./data/memory_db
    :param turns_per_segment: 每段对话轮数，默认 10
    :param embedder: 可选的嵌入器实例，不传则由 MemoryDB 内部创建默认的
                     OllamaEmbedder("verdx/gte-base-zh")
    :return: 包含统计信息的字典
    """
    print(f"📖 正在解析对话文件: {txt_path}")
    turns = parse_dialogue_file(txt_path)
    print(f"   解析完成，共 {len(turns)} 个对话轮次")

    segments = segment_dialogue(turns, turns_per_segment)
    print(f"   按每段 {turns_per_segment} 轮分段，共 {len(segments)} 个片段")

    print(f"💾 正在初始化向量库: {db_path}")
    db = MemoryDB(db_path=db_path, embedder=embedder)

    source_file = os.path.basename(txt_path)
    base_time = datetime.now()
    episode_ids = []

    for idx, segment in enumerate(segments):
        episode_data = build_episode_from_segment(
            segment, idx, source_file, base_time
        )
        db_id = db.add_memory_episode(episode_data)
        episode_ids.append(db_id)
        print(f"   ✅ 片段 {idx + 1}/{len(segments)} 已写入 (db_id={db_id})")

    stats = db.stats()
    db.close()

    summary = {
        "source_file": txt_path,
        "total_turns": len(turns),
        "total_segments": len(segments),
        "episode_ids": episode_ids,
        "db_stats": stats,
    }

    print(f"\n🎉 固化完成！")
    print(f"   总轮次: {len(turns)}")
    print(f"   总片段: {len(segments)}")
    print(f"   向量库路径: {db_path}")
    print(f"   语义向量数: {stats.get('semantic_vectors', 'N/A')}")

    return summary


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="将 txt 对话文件固化为向量记忆库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python txt_to_vectordb.py chat_log.txt
  python txt_to_vectordb.py chat_log.txt --db_path ./my_memory_db
  python txt_to_vectordb.py chat_log.txt --turns_per_segment 5
  python txt_to_vectordb.py chat_log.txt --model gte-base-zh
        """
    )
    parser.add_argument("txt_path", help="txt 对话文件路径")
    parser.add_argument("--db_path", default="./data/memory_db",
                        help="向量库输出目录 (默认: ./data/memory_db)")
    parser.add_argument("--turns_per_segment", type=int, default=10,
                        help="每个记忆片段包含的对话轮数 (默认: 10)")
    parser.add_argument("--model", default="verdx/gte-base-zh",
                        help="Ollama 嵌入模型名称 (默认: verdx/gte-base-zh)")

    args = parser.parse_args()
    embedder = OllamaEmbedder(model_name=args.model)

    txt_to_vectordb(
        txt_path=args.txt_path,
        db_path=args.db_path,
        turns_per_segment=args.turns_per_segment,
        embedder=embedder,
    )


if __name__ == "__main__":
    main()
