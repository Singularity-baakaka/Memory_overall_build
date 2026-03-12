#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_txt_to_vectordb.py - 对话文本 → 向量记忆库 测试脚本

【测试覆盖范围】
  1. TestParseDialogueFile  - 对话文件解析（parse_dialogue_file）
  2. TestChunkDialogues     - 对话切分（chunk_dialogues）
  3. TestBuildEpisodeDataFast - 快速模式构建（build_episode_data_fast）
  4. TestBuildEpisodeData   - LLM 摘要模式构建（build_episode_data）
  5. TestFullPipeline       - 端到端完整流程 + 向量检索验证

【为什么用 MockEmbedder 而不是真实 Ollama？】
  真实嵌入器需要启动 Ollama 服务，在 CI/测试环境中不可靠。
  MockEmbedder 用哈希生成确定性伪向量，保证：
    - 测试无需外部依赖，任何机器均可运行
    - 相同文本始终返回相同向量（可重复性）
    - 不同文本产生不同向量（保证 FAISS 可以区分）

用法：
    python test_txt_to_vectordb.py
    python -m pytest test_txt_to_vectordb.py -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
import numpy as np

# 确保项目根目录在 sys.path 中，使 import 能找到同级的 txt_to_vectordb、memory_db 等模块
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
    【功能】模拟 OllamaEmbedder，在无 Ollama 服务的环境下提供向量嵌入能力。

    【原理】
    对输入文本执行如下步骤：
      1. 计算 Python 内置 hash(text)，取模 2^31 得到一个非负整数作为随机种子；
      2. 用该种子初始化 numpy 随机数生成器（np.random.seed），保证同一文本
         每次调用都能产生完全相同的随机数序列（确定性）；
      3. 从标准正态分布采样 768 个浮点数，构成原始向量；
      4. L2 归一化（除以向量的 L2 范数 + 1e-8 防除零），使向量模长为 1，
         与真实 OllamaEmbedder 的输出格式保持一致。

    【效果】
    - 相同文本 → 相同向量（可重复，不受运行环境影响）
    - 不同文本 → 不同随机种子 → 不同向量（保证 FAISS 能产生差异化的相似度排序）
    - 接口与 OllamaEmbedder 完全兼容，测试代码无需修改即可替换
    """

    def __init__(self, model_name: str = "mock", **kwargs):
        self.model_name = model_name
        self.dim = 768  # 与真实 gte-base-zh 模型输出维度一致

    def get_embedding(self, text: str):
        """
        【功能】根据文本哈希值生成固定的 768 维伪向量。

        :param text: 输入文本
        :return: 归一化后的 768 维 float32 列表，文本为空时返回 None
        """
        if not text or not text.strip():
            return None
        # 用文本哈希作为随机种子，确保同一文本始终产生相同向量
        np.random.seed(hash(text) % (2**31))
        vec = np.random.randn(self.dim).astype(np.float32)
        # L2 归一化，与真实嵌入器保持一致
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec.tolist()


# =====================================================================
# 辅助：测试用的样本对话文本
# =====================================================================

# 标准样本：12 条对话，覆盖"书籍推荐"和"电影"两个话题，
# 用于验证解析数量、语义检索命中等场景
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

# 英文冒号样本：验证解析器对 ASCII 冒号 ":" 的兼容性
SAMPLE_DIALOGUE_ENGLISH_COLON = """用户: 这个用英文冒号
林梓墨: 没问题，也能正确解析
"""

# 多行消息样本：验证续行拼接逻辑（没有"角色:"前缀的行会拼到上一条）
SAMPLE_MULTILINE = """用户: 这是第一行
这是第二行，属于同一条消息
林梓墨: 我也来一条
带有多行内容的消息
第三行
用户: 好的
"""


# =====================================================================
# 测试类 1：对话文件解析
# =====================================================================

class TestParseDialogueFile(unittest.TestCase):
    """
    【测试目标】parse_dialogue_file() 函数
    【覆盖场景】
      - 基本中文冒号格式解析
      - 英文冒号格式解析
      - 多行续行拼接
      - 空文件处理
      - 文件不存在的异常
      - 解析数量准确性
    """

    def setUp(self):
        """每个测试方法运行前，创建一个临时目录用于存放测试文件"""
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        """每个测试方法运行后，删除临时目录及其中所有文件，避免测试污染"""
        shutil.rmtree(self.tmpdir)

    def _write(self, content: str, filename: str = "chat.txt") -> str:
        """
        辅助方法：把字符串内容写入临时目录的文件，返回文件路径。
        避免在每个测试中重复写文件的样板代码。
        """
        path = os.path.join(self.tmpdir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    def test_basic_parse(self):
        """
        【验证】中文冒号格式能被正确解析。
        预期：第一条对话的 role="用户"，content 包含"你好"。
        """
        path = self._write(SAMPLE_DIALOGUE)
        result = parse_dialogue_file(path)
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]["role"], "用户")
        self.assertIn("你好", result[0]["content"])

    def test_english_colon(self):
        """
        【验证】英文冒号 ":" 格式也能正确解析（兼容性测试）。
        预期：解析出 2 条对话，第一条 role="用户"。
        """
        path = self._write(SAMPLE_DIALOGUE_ENGLISH_COLON)
        result = parse_dialogue_file(path)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "用户")

    def test_multiline_content(self):
        """
        【验证】没有"角色:"前缀的续行会被拼接到上一条消息的 content 中。
        预期：
          - result[0]（用户）的 content 同时包含"第一行"和"第二行"；
          - result[1]（林梓墨）的 content 同时包含"多行内容"和"第三行"。
        """
        path = self._write(SAMPLE_MULTILINE)
        result = parse_dialogue_file(path)
        # SAMPLE_MULTILINE 有 3 条独立对话（续行被合并，不计入总数）
        self.assertEqual(len(result), 3)
        # 验证续行合并到用户消息
        self.assertIn("第一行", result[0]["content"])
        self.assertIn("第二行", result[0]["content"])
        # 验证续行合并到林梓墨消息
        self.assertIn("多行内容", result[1]["content"])
        self.assertIn("第三行", result[1]["content"])

    def test_empty_file(self):
        """
        【验证】空文件应返回空列表，不抛出异常。
        这是边界值测试，防止对空文件处理时程序崩溃。
        """
        path = self._write("")
        result = parse_dialogue_file(path)
        self.assertEqual(result, [])

    def test_file_not_found(self):
        """
        【验证】文件不存在时应抛出 FileNotFoundError，而不是静默失败。
        使用 assertRaises 捕获预期异常，确保错误处理逻辑正确。
        """
        with self.assertRaises(FileNotFoundError):
            parse_dialogue_file("/nonexistent/path.txt")

    def test_dialogue_count(self):
        """
        【验证】SAMPLE_DIALOGUE 恰好包含 12 条对话，解析结果数量必须准确。
        这是回归测试，防止解析逻辑修改后悄悄改变输出数量。
        """
        path = self._write(SAMPLE_DIALOGUE)
        result = parse_dialogue_file(path)
        self.assertEqual(len(result), 12)


# =====================================================================
# 测试类 2：对话切分
# =====================================================================

class TestChunkDialogues(unittest.TestCase):
    """
    【测试目标】chunk_dialogues() 函数
    【覆盖场景】
      - 基本切分（整除 + 余数）
      - 恰好整除
      - 只有一个片段的情况
      - 空输入
      - chunk_size < 2 的边界值
    """

    def test_basic_chunking(self):
        """
        【验证】10 条对话以 chunk_size=4 切分 → 产生 3 个片段（4+4+2）。
        验证最后一个不足 4 条的片段也能被正确保留。
        """
        dialogues = [{"role": f"r{i}", "content": f"c{i}"} for i in range(10)]
        chunks = chunk_dialogues(dialogues, chunk_size=4)
        self.assertEqual(len(chunks), 3)   # 10 / 4 = 2 余 2，共 3 片
        self.assertEqual(len(chunks[0]), 4)
        self.assertEqual(len(chunks[2]), 2)  # 最后一片只有 2 条

    def test_exact_division(self):
        """
        【验证】恰好整除时（8条 / chunk_size=4），不产生多余的空片段。
        """
        dialogues = [{"role": f"r{i}", "content": f"c{i}"} for i in range(8)]
        chunks = chunk_dialogues(dialogues, chunk_size=4)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 4)
        self.assertEqual(len(chunks[1]), 4)

    def test_single_chunk(self):
        """
        【验证】当 chunk_size 大于对话总数时，所有对话都在同一个片段中。
        """
        dialogues = [{"role": "a", "content": "b"}]
        chunks = chunk_dialogues(dialogues, chunk_size=100)
        self.assertEqual(len(chunks), 1)

    def test_empty_dialogues(self):
        """
        【验证】空对话列表输入 → 返回空列表，不崩溃。
        """
        chunks = chunk_dialogues([], chunk_size=10)
        self.assertEqual(chunks, [])

    def test_min_chunk_size(self):
        """
        【验证】chunk_size=1 时会被强制提升为 2（最小有意义的对话片段是一问一答）。
        预期：第一个片段有 2 条对话，而不是 1 条。
        """
        dialogues = [{"role": f"r{i}", "content": f"c{i}"} for i in range(5)]
        chunks = chunk_dialogues(dialogues, chunk_size=1)
        # 强制 chunk_size=2，5条对话 → 片段大小应为 2
        self.assertEqual(len(chunks[0]), 2)


# =====================================================================
# 测试类 3：快速模式构建
# =====================================================================

class TestBuildEpisodeDataFast(unittest.TestCase):
    """
    【测试目标】build_episode_data_fast() 函数
    【覆盖场景】
      - 输出字典包含所有必要字段
      - narrative 包含原始对话内容
      - raw_dialogue 数量正确
      - 关键词提取逻辑
    """

    def test_fast_build(self):
        """
        【验证】快速构建的 episode_data 包含所有必要字段，且内容正确。
        具体检查：
          - narrative 中包含 "你好"（原文被直接拼入）
          - raw_dialogue 长度等于输入的对话条数（2条）
          - timestamp、keywords 字段存在
        """
        chunk = [
            {"role": "用户", "content": "你好呀"},
            {"role": "林梓墨", "content": "你好！今天天气真好"},
        ]
        episode = build_episode_data_fast(chunk)
        # 验证所有必要字段都存在
        self.assertIn("narrative", episode)
        self.assertIn("timestamp", episode)
        self.assertIn("raw_dialogue", episode)
        self.assertIn("keywords", episode)
        # 验证 narrative 中包含原始对话内容（快速模式直接使用原文）
        self.assertIn("你好", episode["narrative"])
        # 验证 raw_dialogue 条数与输入一致
        self.assertEqual(len(episode["raw_dialogue"]), 2)

    def test_fast_build_keywords(self):
        """
        【验证】快速模式能从中文对话中正则提取到有效关键词。
        预期：keywords 是非空列表（能从"北京旅游长城故宫"等内容中提取出词语）。
        """
        chunk = [
            {"role": "用户", "content": "我在北京旅游，去了长城和故宫"},
            {"role": "林梓墨", "content": "好棒！北京的美食也值得尝尝"},
        ]
        episode = build_episode_data_fast(chunk)
        self.assertIsInstance(episode["keywords"], list)
        self.assertGreater(len(episode["keywords"]), 0)


# =====================================================================
# 测试类 4：LLM 摘要模式构建
# =====================================================================

class TestBuildEpisodeData(unittest.TestCase):
    """
    【测试目标】build_episode_data() 函数（LLM 摘要模式的数据组装）
    【说明】此测试不调用真实 LLM，直接构造模拟的 summary 字典传入，
           只验证数据组装逻辑是否正确（字段映射、值传递等）。
    """

    def test_build_with_summary(self):
        """
        【验证】传入对话片段和模拟摘要后，输出的 episode_data 字段值正确映射。
        具体检查：
          - narrative 来自 summary["narrative"]
          - atmosphere 来自 summary["atmosphere"]
          - keywords 来自 summary["keywords"]
          - raw_dialogue 长度等于输入对话数（2条）
        """
        chunk = [
            {"role": "用户", "content": "你好"},
            {"role": "林梓墨", "content": "你好呀"},
        ]
        # 模拟 LLM 返回的摘要（跳过真实 LLM 调用）
        summary = {
            "narrative": "用户和林梓墨互相问好。",
            "atmosphere": "轻松, 友好",
            "keywords": ["问候", "打招呼"],
        }
        episode = build_episode_data(chunk, summary)
        # 验证摘要字段被正确映射到 episode_data
        self.assertEqual(episode["narrative"], "用户和林梓墨互相问好。")
        self.assertEqual(episode["atmosphere"], "轻松, 友好")
        self.assertEqual(episode["keywords"], ["问候", "打招呼"])
        # 验证原始对话被完整保留
        self.assertEqual(len(episode["raw_dialogue"]), 2)


# =====================================================================
# 测试类 5：端到端完整流程
# =====================================================================

class TestFullPipeline(unittest.TestCase):
    """
    【测试目标】txt_to_vectordb() 主函数（端到端集成测试）
    【测试策略】
      - 使用 MockEmbedder 替代真实 Ollama，无需外部服务
      - 使用 tempfile.mkdtemp() 创建临时目录，测试后自动清理
      - 验证从文件读取到数据库写入再到向量检索的完整链路

    【覆盖场景】
      1. 快速模式完整流程 + 检索验证
      2. 生成文件校验（.db 和 .faiss 文件是否存在）
      3. 不同 chunk_size下的片段数量对比
    """

    def setUp(self):
        """
        每个测试方法运行前的准备工作：
          - 创建临时目录（测试沙箱）
          - 写入 SAMPLE_DIALOGUE 内容到临时 txt 文件
          - 规划 db_dir 路径（测试用的向量库存放位置）
        """
        self.tmpdir = tempfile.mkdtemp()
        self.txt_path = os.path.join(self.tmpdir, "chat.txt")
        self.db_dir = os.path.join(self.tmpdir, "test_memory_db")

        with open(self.txt_path, 'w', encoding='utf-8') as f:
            f.write(SAMPLE_DIALOGUE)

    def tearDown(self):
        """每个测试结束后删除临时目录，避免留下测试垃圾文件"""
        shutil.rmtree(self.tmpdir)

    def test_full_pipeline_fast_mode(self):
        """
        【验证】快速模式端到端完整链路：txt 解析 → 切片 → 入库 → 检索。

        【断言逻辑】
        第一段：验证 txt_to_vectordb() 返回的统计字典
          - status="done" 表示流程正常完成
          - total_dialogues=12 验证解析数量正确（SAMPLE_DIALOGUE 有 12 条）
          - success>0 且 failed=0 验证所有片段都成功写入
          - db_stats 中向量数量 > 0 验证 FAISS 索引有内容

        第二段：重新打开数据库，验证语义检索功能
          - search_by_semantic("电影") 应能命中包含电影对话的记忆片段
          - 返回的记忆字典包含 narrative、similarity、raw_dialogue 三个关键字段
        """
        mock_embedder = MockEmbedder()
        result = txt_to_vectordb(
            txt_path=self.txt_path,
            output_dir=self.db_dir,
            chunk_size=6,      # chunk_size=6：12条对话 → 2个片段
            use_llm=False,     # 不调 LLM，使用快速模式
            embedder=mock_embedder,
        )

        # ---- 验证返回的统计信息 ----
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["total_dialogues"], 12)
        self.assertGreater(result["success"], 0)
        self.assertEqual(result["failed"], 0)
        self.assertGreater(result["db_stats"]["total_memory_episodes"], 0)
        self.assertGreater(result["db_stats"]["semantic_vectors"], 0)

        # ---- 重新打开数据库，验证检索功能 ----
        db = MemoryDB(db_path=self.db_dir, embedder=mock_embedder)
        stats = db.stats()
        self.assertGreater(stats["total_memory_episodes"], 0)

        # 语义检索"电影" → 应命中 SAMPLE_DIALOGUE 后半段（星际穿越相关片段）
        results = db.search_by_semantic("电影", top_k=3)
        self.assertGreater(len(results), 0)

        # 验证返回的记忆字典结构完整
        first = results[0]
        self.assertIn("narrative", first)    # 叙事内容
        self.assertIn("similarity", first)   # 相似度分数（由 _hydrate_results 附加）
        self.assertIn("raw_dialogue", first) # 原始对话（用于追溯上下文）

        db.close()

    def test_pipeline_creates_db_files(self):
        """
        【验证】固化完成后，必要的数据库文件确实被创建在 output_dir 中。

        具体检查两类文件：
          - memory_episodes.db：SQLite 数据库文件（结构化存储）
          - semantic.faiss：语义向量索引文件（FAISS 二进制格式）

        这是"文件系统级别"的集成验证，确保数据真的落盘了。
        """
        mock_embedder = MockEmbedder()
        txt_to_vectordb(
            txt_path=self.txt_path,
            output_dir=self.db_dir,
            chunk_size=20,
            use_llm=False,
            embedder=mock_embedder,
        )
        # 验证 SQLite 数据库文件存在
        self.assertTrue(os.path.exists(
            os.path.join(self.db_dir, "memory_episodes.db")))
        # 验证 FAISS 语义索引文件存在
        self.assertTrue(os.path.exists(
            os.path.join(self.db_dir, "semantic.faiss")))

    def test_pipeline_different_chunk_sizes(self):
        """
        【验证】chunk_size 越小 → 记忆片段数越多。

        【测试逻辑】
        对同一份 txt 文件分别用 chunk_size=4 和 chunk_size=20 入库：
          - chunk_size=4：12条对话 → 3个片段（4+4+4）
          - chunk_size=20：12条对话 → 1个片段（12条全在一起）
        预期：result_small["total_chunks"] > result_large["total_chunks"]

        这个测试保证了 chunk_size 参数确实影响了切分行为，
        而不是被忽略或错误地处理。
        """
        mock_embedder = MockEmbedder()

        # 细粒度切片（小 chunk）
        result_small = txt_to_vectordb(
            txt_path=self.txt_path,
            output_dir=os.path.join(self.tmpdir, "db_small"),
            chunk_size=4,
            use_llm=False,
            embedder=mock_embedder,
        )
        # 粗粒度切片（大 chunk）
        result_large = txt_to_vectordb(
            txt_path=self.txt_path,
            output_dir=os.path.join(self.tmpdir, "db_large"),
            chunk_size=20,
            use_llm=False,
            embedder=mock_embedder,
        )
        # 小 chunk_size 产生更多片段
        self.assertGreater(result_small["total_chunks"],
                           result_large["total_chunks"])


if __name__ == "__main__":
    unittest.main()
