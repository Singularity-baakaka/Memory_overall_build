#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_txt_to_vectordb.py - 测试脚本：验证对话文本固化向量库的完整流程

测试内容：
1. 对话文件解析（多种格式）
2. 分段逻辑
3. 关键词提取
4. episode 构建
5. 端到端写入与检索验证（使用 mock embedder 避免依赖 Ollama 服务）

用法：
  python test_txt_to_vectordb.py
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from typing import List, Optional

from txt_to_vectordb import (
    parse_dialogue_file,
    segment_dialogue,
    extract_simple_keywords,
    build_episode_from_segment,
    txt_to_vectordb,
)
from memory_db import MemoryDB


# =====================================================================
# Mock Embedder：用确定性的伪向量替代真实 Ollama API 调用
# =====================================================================

class MockEmbedder:
    """
    模拟嵌入器：根据文本内容的哈希值生成确定性伪向量。
    相同文本始终返回相同向量，不同文本返回不同向量。
    用于离线测试，无需启动 Ollama 服务。
    """

    def __init__(self, model_name: str = "mock-model",
                 api_url: str = "", retries: int = 1):
        self.model_name = model_name
        self.api_url = api_url
        self.retries = retries

    def get_embedding(self, text: str) -> Optional[List[float]]:
        if not text or not text.strip():
            return None
        # 基于文本哈希生成确定性的 768 维伪向量
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(768).astype(np.float32)
        # L2 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


# =====================================================================
# 辅助函数
# =====================================================================

def create_temp_dialogue_file(content: str) -> str:
    """创建临时对话文件，返回文件路径"""
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="test_dialogue_")
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


# =====================================================================
# 测试用例
# =====================================================================

class TestParseDialogueFile(unittest.TestCase):
    """测试对话文件解析"""

    def test_basic_chinese_colon(self):
        """测试中文冒号格式"""
        content = "用户：你好\n林梓墨：你好呀！\n用户：今天天气怎么样\n林梓墨：今天天气很好呢\n"
        path = create_temp_dialogue_file(content)
        try:
            turns = parse_dialogue_file(path)
            self.assertEqual(len(turns), 4)
            self.assertEqual(turns[0]["role"], "用户")
            self.assertEqual(turns[0]["content"], "你好")
            self.assertEqual(turns[1]["role"], "林梓墨")
            self.assertEqual(turns[1]["content"], "你好呀！")
        finally:
            os.unlink(path)

    def test_english_colon(self):
        """测试英文冒号格式"""
        content = "User: Hello\nAssistant: Hi there!\nUser: How are you?\nAssistant: I'm great!\n"
        path = create_temp_dialogue_file(content)
        try:
            turns = parse_dialogue_file(path)
            self.assertEqual(len(turns), 4)
            self.assertEqual(turns[0]["role"], "User")
            self.assertEqual(turns[2]["content"], "How are you?")
        finally:
            os.unlink(path)

    def test_multiline_content(self):
        """测试多行续写内容"""
        content = "用户: 你好\n请问今天天气怎么样？\n我想出去玩\n林梓墨: 今天天气不错哦\n"
        path = create_temp_dialogue_file(content)
        try:
            turns = parse_dialogue_file(path)
            self.assertEqual(len(turns), 2)
            self.assertIn("请问今天天气怎么样？", turns[0]["content"])
            self.assertIn("我想出去玩", turns[0]["content"])
        finally:
            os.unlink(path)

    def test_blank_line_separation(self):
        """测试空行分隔的对话"""
        content = "用户: 你好\n\n林梓墨: 你好\n\n用户: 再见\n\n林梓墨: 再见\n"
        path = create_temp_dialogue_file(content)
        try:
            turns = parse_dialogue_file(path)
            self.assertEqual(len(turns), 4)
        finally:
            os.unlink(path)

    def test_empty_file_raises(self):
        """测试空文件抛出异常"""
        path = create_temp_dialogue_file("")
        try:
            with self.assertRaises(ValueError):
                parse_dialogue_file(path)
        finally:
            os.unlink(path)

    def test_nonexistent_file_raises(self):
        """测试文件不存在抛出异常"""
        with self.assertRaises(FileNotFoundError):
            parse_dialogue_file("/tmp/nonexistent_dialogue_12345.txt")


class TestSegmentDialogue(unittest.TestCase):
    """测试对话分段"""

    def _make_turns(self, n: int) -> list:
        return [{"role": f"r{i}", "content": f"content {i}"} for i in range(n)]

    def test_exact_split(self):
        """测试刚好整除的分段"""
        turns = self._make_turns(10)
        segments = segment_dialogue(turns, turns_per_segment=5)
        self.assertEqual(len(segments), 2)
        self.assertEqual(len(segments[0]), 5)
        self.assertEqual(len(segments[1]), 5)

    def test_remainder_segment(self):
        """测试有余数的分段"""
        turns = self._make_turns(7)
        segments = segment_dialogue(turns, turns_per_segment=3)
        self.assertEqual(len(segments), 3)
        self.assertEqual(len(segments[2]), 1)

    def test_single_segment(self):
        """测试不需要分段的情况"""
        turns = self._make_turns(3)
        segments = segment_dialogue(turns, turns_per_segment=10)
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(segments[0]), 3)

    def test_minimum_segment_size(self):
        """测试最小分段参数"""
        turns = self._make_turns(3)
        segments = segment_dialogue(turns, turns_per_segment=0)
        self.assertEqual(len(segments), 3)


class TestExtractKeywords(unittest.TestCase):
    """测试关键词提取"""

    def test_chinese_keywords(self):
        """测试中文关键词提取"""
        text = "今天天气很好，我们去公园散步了。公园里有很多花，天气真的很好。"
        keywords = extract_simple_keywords(text)
        self.assertTrue(len(keywords) > 0)
        # "天气" 和 "公园" 应该是高频词
        self.assertIn("天气", keywords)
        self.assertIn("公园", keywords)

    def test_english_keywords(self):
        """测试英文关键词提取"""
        text = "The weather is great today. Let's go to the park. The park has many flowers."
        keywords = extract_simple_keywords(text)
        self.assertTrue(len(keywords) > 0)

    def test_max_keywords_limit(self):
        """测试关键词数量限制"""
        text = "苹果 香蕉 橘子 葡萄 西瓜 芒果 草莓 蓝莓 樱桃 菠萝"
        keywords = extract_simple_keywords(text, max_keywords=3)
        self.assertTrue(len(keywords) <= 3)


class TestBuildEpisode(unittest.TestCase):
    """测试 episode 构建"""

    def test_basic_build(self):
        """测试基本 episode 构建"""
        segment = [
            {"role": "用户", "content": "你好"},
            {"role": "AI", "content": "你好呀！"},
        ]
        episode = build_episode_from_segment(segment, 0, "test.txt")
        self.assertIn("timestamp", episode)
        self.assertIn("narrative", episode)
        self.assertIn("raw_dialogue", episode)
        self.assertIn("keywords", episode)
        self.assertIn("用户: 你好", episode["narrative"])
        self.assertIn("AI: 你好呀！", episode["narrative"])
        self.assertEqual(len(episode["raw_dialogue"]), 2)

    def test_keywords_extracted(self):
        """测试 episode 中包含提取的关键词"""
        segment = [
            {"role": "用户", "content": "我想去北京旅游"},
            {"role": "AI", "content": "北京是个好地方，有很多景点"},
        ]
        episode = build_episode_from_segment(segment, 0, "test.txt")
        self.assertIsInstance(episode["keywords"], list)


class TestEndToEnd(unittest.TestCase):
    """端到端测试：完整的 txt → 向量库 → 检索 流程"""

    def setUp(self):
        """创建临时目录和测试数据"""
        self.test_dir = tempfile.mkdtemp(prefix="test_vectordb_")
        self.db_path = os.path.join(self.test_dir, "test_db")

        # 创建示例对话文件
        self.dialogue_content = (
            "用户: 你好，我叫小明\n"
            "林梓墨: 你好小明！很高兴认识你\n"
            "用户: 今天天气很好，我们去公园散步吧\n"
            "林梓墨: 好呀，公园里的樱花应该开了\n"
            "用户: 你喜欢什么花\n"
            "林梓墨: 我最喜欢樱花和玫瑰\n"
            "用户: 昨天我去看了一部电影\n"
            "林梓墨: 是什么电影呀\n"
            "用户: 一部科幻片，讲的是时间旅行\n"
            "林梓墨: 听起来很有意思！我也想看\n"
            "用户: 下次我们一起去看吧\n"
            "林梓墨: 好的，一言为定\n"
        )
        self.dialogue_path = os.path.join(self.test_dir, "test_dialogue.txt")
        with open(self.dialogue_path, 'w', encoding='utf-8') as f:
            f.write(self.dialogue_content)

    def tearDown(self):
        """清理临时文件"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_full_pipeline(self):
        """测试完整流程：解析 → 分段 → 写入 → 统计"""
        mock_embedder = MockEmbedder()

        summary = txt_to_vectordb(
            txt_path=self.dialogue_path,
            db_path=self.db_path,
            turns_per_segment=4,
            embedder=mock_embedder,
        )

        # 验证统计信息
        self.assertEqual(summary["total_turns"], 12)
        self.assertEqual(summary["total_segments"], 3)
        self.assertEqual(len(summary["episode_ids"]), 3)

        # 验证数据库文件已创建
        db_file = os.path.join(self.db_path, "memory_episodes.db")
        self.assertTrue(os.path.exists(db_file))

        # 验证 FAISS 索引文件已创建
        self.assertTrue(os.path.exists(
            os.path.join(self.db_path, "semantic.faiss")))

    def test_search_after_import(self):
        """测试写入后可以通过检索找回记忆"""
        mock_embedder = MockEmbedder()

        txt_to_vectordb(
            txt_path=self.dialogue_path,
            db_path=self.db_path,
            turns_per_segment=6,
            embedder=mock_embedder,
        )

        # 重新打开数据库进行检索
        db = MemoryDB(db_path=self.db_path, embedder=mock_embedder)

        # 验证记忆总数
        stats = db.stats()
        self.assertEqual(stats["total_memory_episodes"], 2)
        self.assertTrue(stats["semantic_vectors"] > 0)

        # 通过 ID 获取记忆
        episode = db.get_episode_by_id(1)
        self.assertIsNotNone(episode)
        self.assertIn("narrative", episode)

        # 验证 narrative 内容包含原始对话
        self.assertIn("用户", episode["narrative"])

        # 通过语义搜索
        results = db.search_by_semantic("公园散步", top_k=2)
        self.assertTrue(len(results) > 0)

        db.close()

    def test_single_turn_per_segment(self):
        """测试极端情况：每段只有 1 轮"""
        mock_embedder = MockEmbedder()

        summary = txt_to_vectordb(
            txt_path=self.dialogue_path,
            db_path=self.db_path,
            turns_per_segment=1,
            embedder=mock_embedder,
        )

        self.assertEqual(summary["total_segments"], 12)
        self.assertEqual(len(summary["episode_ids"]), 12)

    def test_large_segment(self):
        """测试不分段的情况（段大小 > 总轮数）"""
        mock_embedder = MockEmbedder()

        summary = txt_to_vectordb(
            txt_path=self.dialogue_path,
            db_path=self.db_path,
            turns_per_segment=100,
            embedder=mock_embedder,
        )

        self.assertEqual(summary["total_segments"], 1)
        self.assertEqual(len(summary["episode_ids"]), 1)


if __name__ == "__main__":
    print("=" * 60)
    print("  对话文本固化向量库 - 测试脚本")
    print("=" * 60)
    unittest.main(verbosity=2)
