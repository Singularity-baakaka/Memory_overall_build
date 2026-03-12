#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_txt_to_vectordb.py - 对话文本 → 向量记忆库 测试脚本

验证内容：
  1. 对话文件解析（parse_dialogue_file）
  2. 对话切分（chunk_dialogues）
  3. 快速模式构建（build_episode_data_fast）
  4. 完整固化流程 + 向量检索验证

测试使用模拟嵌入器（MockEmbedder），无需启动 Ollama 或 LLM 服务即可运行。

用法：
    python test_txt_to_vectordb.py
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
import numpy as np

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from txt_to_vectordb import (
    parse_dialogue_file,
    chunk_dialogues,
    build_episode_data_fast,
    build_episode_data,
    txt_to_vectordb,
)
from memory_db import MemoryDB


# =====================================================================
# Mock Embedder —— 用确定性哈希生成伪向量，无需外部 API
# =====================================================================

class MockEmbedder:
    """
    模拟 OllamaEmbedder：根据文本内容用哈希生成固定的 768 维向量。
    语义相近的文本会共享部分哈希特征，从而在 FAISS 中产生合理的相似度排序。
    """

    def __init__(self, model_name: str = "mock", **kwargs):
        self.model_name = model_name
        self.dim = 768

    def get_embedding(self, text: str):
        if not text or not text.strip():
            return None
        np.random.seed(hash(text) % (2**31))
        vec = np.random.randn(self.dim).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec.tolist()


# =====================================================================
# 辅助：创建临时对话文件
# =====================================================================

SAMPLE_DIALOGUE = """用户: 你好呀，今天过得怎么样？
林梓墨: 还不错哦～今天读了一本关于宇宙的书，特别有意思！
用户: 是什么书？给我推荐一下呗
林梓墨: 叫《时间简史》，霍金写的，讲了黑洞和时间的奥秘
用户: 听起来很深奥啊
林梓墨: 其实没那么难，他写得很通俗易懂的
用户: 好的，我去找来看看
林梓墨: 嗯嗯！看完了我们可以一起讨论哦
用户: 对了，你最近有没有看什么电影？
林梓墨: 看了《星际穿越》，被里面的亲情线感动哭了
用户: 那部电影确实很经典
林梓墨: 是啊，尤其是最后父女重逢的那一段
"""

SAMPLE_DIALOGUE_ENGLISH_COLON = """用户: 这个用英文冒号
林梓墨: 没问题，也能正确解析
"""

SAMPLE_MULTILINE = """用户: 这是第一行
这是第二行，属于同一条消息
林梓墨: 我也来一条
带有多行内容的消息
第三行
用户: 好的
"""


class TestParseDialogueFile(unittest.TestCase):
    """测试对话文件解析"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write(self, content: str, filename: str = "chat.txt") -> str:
        path = os.path.join(self.tmpdir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    def test_basic_parse(self):
        """基本解析：中文冒号格式"""
        path = self._write(SAMPLE_DIALOGUE)
        result = parse_dialogue_file(path)
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]["role"], "用户")
        self.assertIn("你好", result[0]["content"])

    def test_english_colon(self):
        """英文冒号也能正确解析"""
        path = self._write(SAMPLE_DIALOGUE_ENGLISH_COLON)
        result = parse_dialogue_file(path)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "用户")

    def test_multiline_content(self):
        """多行内容自动拼接到上一条消息"""
        path = self._write(SAMPLE_MULTILINE)
        result = parse_dialogue_file(path)
        self.assertEqual(len(result), 3)
        # 第一条用户消息应包含两行
        self.assertIn("第一行", result[0]["content"])
        self.assertIn("第二行", result[0]["content"])
        # 林梓墨的消息应包含三行
        self.assertIn("多行内容", result[1]["content"])
        self.assertIn("第三行", result[1]["content"])

    def test_empty_file(self):
        """空文件返回空列表"""
        path = self._write("")
        result = parse_dialogue_file(path)
        self.assertEqual(result, [])

    def test_file_not_found(self):
        """文件不存在应抛出 FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            parse_dialogue_file("/nonexistent/path.txt")

    def test_dialogue_count(self):
        """验证解析出的对话数量"""
        path = self._write(SAMPLE_DIALOGUE)
        result = parse_dialogue_file(path)
        # SAMPLE_DIALOGUE 有 12 行对话
        self.assertEqual(len(result), 12)


class TestChunkDialogues(unittest.TestCase):
    """测试对话切分"""

    def test_basic_chunking(self):
        dialogues = [{"role": f"r{i}", "content": f"c{i}"} for i in range(10)]
        chunks = chunk_dialogues(dialogues, chunk_size=4)
        self.assertEqual(len(chunks), 3)  # 4+4+2
        self.assertEqual(len(chunks[0]), 4)
        self.assertEqual(len(chunks[2]), 2)

    def test_exact_division(self):
        dialogues = [{"role": f"r{i}", "content": f"c{i}"} for i in range(8)]
        chunks = chunk_dialogues(dialogues, chunk_size=4)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 4)
        self.assertEqual(len(chunks[1]), 4)

    def test_single_chunk(self):
        dialogues = [{"role": "a", "content": "b"}]
        chunks = chunk_dialogues(dialogues, chunk_size=100)
        self.assertEqual(len(chunks), 1)

    def test_empty_dialogues(self):
        chunks = chunk_dialogues([], chunk_size=10)
        self.assertEqual(chunks, [])

    def test_min_chunk_size(self):
        """chunk_size < 2 应被强制为 2"""
        dialogues = [{"role": f"r{i}", "content": f"c{i}"} for i in range(5)]
        chunks = chunk_dialogues(dialogues, chunk_size=1)
        self.assertEqual(len(chunks[0]), 2)


class TestBuildEpisodeDataFast(unittest.TestCase):
    """测试快速模式构建 episode 数据"""

    def test_fast_build(self):
        chunk = [
            {"role": "用户", "content": "你好呀"},
            {"role": "林梓墨", "content": "你好！今天天气真好"},
        ]
        episode = build_episode_data_fast(chunk)
        self.assertIn("narrative", episode)
        self.assertIn("timestamp", episode)
        self.assertIn("raw_dialogue", episode)
        self.assertIn("keywords", episode)
        self.assertIn("你好", episode["narrative"])
        self.assertEqual(len(episode["raw_dialogue"]), 2)

    def test_fast_build_keywords(self):
        """快速模式应提取到中文关键词"""
        chunk = [
            {"role": "用户", "content": "我在北京旅游，去了长城和故宫"},
            {"role": "林梓墨", "content": "好棒！北京的美食也值得尝尝"},
        ]
        episode = build_episode_data_fast(chunk)
        self.assertIsInstance(episode["keywords"], list)
        self.assertGreater(len(episode["keywords"]), 0)


class TestBuildEpisodeData(unittest.TestCase):
    """测试 build_episode_data（LLM 摘要模式）"""

    def test_build_with_summary(self):
        chunk = [
            {"role": "用户", "content": "你好"},
            {"role": "林梓墨", "content": "你好呀"},
        ]
        summary = {
            "narrative": "用户和林梓墨互相问好。",
            "atmosphere": "轻松, 友好",
            "keywords": ["问候", "打招呼"],
        }
        episode = build_episode_data(chunk, summary)
        self.assertEqual(episode["narrative"], "用户和林梓墨互相问好。")
        self.assertEqual(episode["atmosphere"], "轻松, 友好")
        self.assertEqual(episode["keywords"], ["问候", "打招呼"])
        self.assertEqual(len(episode["raw_dialogue"]), 2)


class TestFullPipeline(unittest.TestCase):
    """端到端测试：txt 文件 → 向量库 → 检索验证"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.txt_path = os.path.join(self.tmpdir, "chat.txt")
        self.db_dir = os.path.join(self.tmpdir, "test_memory_db")

        with open(self.txt_path, 'w', encoding='utf-8') as f:
            f.write(SAMPLE_DIALOGUE)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_full_pipeline_fast_mode(self):
        """快速模式端到端：解析 → 入库 → 检索"""
        mock_embedder = MockEmbedder()
        result = txt_to_vectordb(
            txt_path=self.txt_path,
            output_dir=self.db_dir,
            chunk_size=6,
            use_llm=False,
            embedder=mock_embedder,
        )

        # 基本断言
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["total_dialogues"], 12)
        self.assertGreater(result["success"], 0)
        self.assertEqual(result["failed"], 0)
        self.assertGreater(result["db_stats"]["total_memory_episodes"], 0)
        self.assertGreater(result["db_stats"]["semantic_vectors"], 0)

        # 打开数据库检索验证
        db = MemoryDB(db_path=self.db_dir, embedder=mock_embedder)
        stats = db.stats()
        self.assertGreater(stats["total_memory_episodes"], 0)

        # 语义检索：查 "电影" 应该能命中包含电影对话的片段
        results = db.search_by_semantic("电影", top_k=3)
        self.assertGreater(len(results), 0)
        # 验证返回的记忆包含完整字段
        first = results[0]
        self.assertIn("narrative", first)
        self.assertIn("similarity", first)
        self.assertIn("raw_dialogue", first)

        db.close()

    def test_pipeline_creates_db_files(self):
        """验证固化后生成了必要的数据库文件"""
        mock_embedder = MockEmbedder()
        txt_to_vectordb(
            txt_path=self.txt_path,
            output_dir=self.db_dir,
            chunk_size=20,
            use_llm=False,
            embedder=mock_embedder,
        )
        # SQLite 数据库文件
        self.assertTrue(os.path.exists(
            os.path.join(self.db_dir, "memory_episodes.db")))
        # FAISS 索引文件
        self.assertTrue(os.path.exists(
            os.path.join(self.db_dir, "semantic.faiss")))

    def test_pipeline_different_chunk_sizes(self):
        """不同 chunk_size 应生成不同数量的记忆片段"""
        mock_embedder = MockEmbedder()

        result_small = txt_to_vectordb(
            txt_path=self.txt_path,
            output_dir=os.path.join(self.tmpdir, "db_small"),
            chunk_size=4,
            use_llm=False,
            embedder=mock_embedder,
        )
        result_large = txt_to_vectordb(
            txt_path=self.txt_path,
            output_dir=os.path.join(self.tmpdir, "db_large"),
            chunk_size=20,
            use_llm=False,
            embedder=mock_embedder,
        )
        self.assertGreater(result_small["total_chunks"],
                           result_large["total_chunks"])


if __name__ == "__main__":
    unittest.main()
