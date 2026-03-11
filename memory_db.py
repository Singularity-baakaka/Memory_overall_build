#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
memory_db.py - 记忆数据库 (DAO Layer)
任务：只负责对记忆切片的读写与 FAISS 向量库维护。
结构：每个记忆切片包含 3 个维度的向量索引（整体语义、氛围、关键词）。
无任何高级业务逻辑，时间戳和重要性由外部传入。
"""

import sqlite3
import json
import numpy as np
import faiss
import os
from typing import List, Dict, Optional, Tuple, Union 

from utils import OllamaEmbedder, normalize_vector

VECTOR_DIM = 768

class MemoryDB:
    """
    记忆数据库 DAO (Data Access Object) 层。

    原理：构建一个多维度的向量记忆库，将每段记忆（episode）同时存储于
    SQLite（结构化数据）和 FAISS（向量索引）中，实现混合检索。
    每条记忆切片包含三个维度的向量索引：
    - 语义向量（narrative）：整体叙事内容的语义嵌入
    - 氛围向量（atmosphere）：情感/氛围标签的语义嵌入
    - 关键词向量（keywords）：各关键词的独立语义嵌入（一对多）
    通过三维度加权融合检索，可以从不同角度匹配和召回相关记忆。
    """

    def __init__(self, db_path: str = "./data/memory_db", embedder: Optional[OllamaEmbedder] = None):
        """
        初始化 MemoryDB 实例，创建或连接数据库并加载向量索引。

        原理：
        1. 确保数据目录存在（os.makedirs）；
        2. 初始化或复用外部传入的 OllamaEmbedder 嵌入器；
        3. 连接 SQLite 数据库文件（memory_episodes.db），设置 check_same_thread=False 允许跨线程访问，row_factory=sqlite3.Row 使查询结果可按列名访问；
        4. 创建数据表结构（若不存在）；
        5. 初始化三个 FAISS 向量索引的文件路径；
        6. 从磁盘加载已有的 FAISS 索引（若存在），实现数据持久化。

        :param db_path: 数据存储目录路径
        :param embedder: 可选的嵌入器实例，不传则使用默认的 gte-base-zh 模型
        """
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)

        self.embedder = embedder if embedder else OllamaEmbedder("verdx/gte-base-zh")

        # 连接 SQLite (使用一张主表)
        self.conn = sqlite3.connect(
            os.path.join(db_path, "memory_episodes.db"), 
            check_same_thread=False 
        )
        self.conn.row_factory = sqlite3.Row
        self._init_tables() 

        # 初始化并加载 FAISS 索引
        self._init_faiss_indices()
        self._load_faiss_indices()

    def _init_tables(self):
        """
        初始化 SQLite 数据库表结构。

        原理：执行 CREATE TABLE IF NOT EXISTS 语句创建 memory_episodes 主表。
        使用 IF NOT EXISTS 保证幂等性——表已存在时不会重复创建或覆盖数据。
        表结构设计：
        - id: 自增主键，作为每条记忆的唯一标识，也用于 FAISS 映射表的关联
        - timestamp: 外部传入的时间戳，记录事件发生的时间
        - narrative: 记忆的整体叙事描述，是语义检索的主要内容
        - raw_dialogue: 原始对话的 JSON 备份，用于追溯完整对话上下文
        - atmosphere: 氛围/情感标签文本，用于氛围维度的检索
        - keywords: 关键词 JSON 列表，用于关键词维度的精准检索
        - importance: 重要性权重（0~1），影响混合检索的最终排序
        - created_at: 自动记录的数据库写入时间
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,          -- 外部传入的时间戳
                narrative TEXT NOT NULL,          -- 整体描述/叙事
                raw_dialogue TEXT,                -- 原始对话备份(可选)
                atmosphere TEXT,                  -- 氛围标签
                keywords TEXT,                    -- 关键词 JSON 列表
                importance REAL DEFAULT 0.5,      -- 重要性权重
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def _init_faiss_indices(self):
        """
        初始化三大 FAISS 向量索引的文件路径配置。

        原理：为语义（semantic）、氛围（atmosphere）、关键词（keyword）三个检索维度
        分别规划 FAISS 索引文件（.faiss）和映射表文件（_map.json）的存储路径。
        FAISS 索引文件存储向量数据及其内部 ID，映射表 JSON 文件维护
        "FAISS 内部 ID → SQLite 数据库 ID" 的映射关系。
        此方法仅设置路径，不实际创建或加载索引。

        三个维度的设计目的：
        - semantic: 通过叙事文本的整体语义进行模糊匹配
        - atmosphere: 通过情感氛围标签进行情境匹配
        - keyword: 通过具体关键词进行精准匹配
        """
        self.dim = VECTOR_DIM
        
        # 文件路径
        self.semantic_index_path = os.path.join(self.db_path, "semantic.faiss")
        self.atmosphere_index_path = os.path.join(self.db_path, "atmosphere.faiss")
        self.keyword_index_path = os.path.join(self.db_path, "keyword.faiss")
        
        self.semantic_map_path = os.path.join(self.db_path, "semantic_map.json")
        self.atmosphere_map_path = os.path.join(self.db_path, "atmosphere_map.json")
        self.keyword_map_path = os.path.join(self.db_path, "keyword_map.json")

    def _load_faiss_indices(self):
        """
        从磁盘加载 FAISS 索引及其映射表（faiss_id → db_id）。

        原理：程序启动时需要恢复之前持久化的向量索引状态。
        对每个维度的索引：
        - 若磁盘上存在 .faiss 文件，则用 faiss.read_index 加载；
        - 若不存在（首次运行），则创建空的 IndexFlatIP 索引。
          IndexFlatIP 使用内积（Inner Product）作为距离度量，
          当输入向量经过 L2 归一化后，内积等价于余弦相似度。
        映射表（JSON dict）记录 FAISS 内部索引位置到数据库记录 ID 的对应关系，
        因为 FAISS 内部使用连续整数编号，与 SQLite 自增主键可能不一致。
        """
        def load_index(path):
            """加载单个 FAISS 索引文件，不存在则创建空索引"""
            return faiss.read_index(path) if os.path.exists(path) else faiss.IndexFlatIP(self.dim)

        def load_map(path):
            """加载单个映射表 JSON 文件，不存在则返回空字典"""
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}

        self.semantic_index = load_index(self.semantic_index_path)
        self.atmosphere_index = load_index(self.atmosphere_index_path)
        self.keyword_index = load_index(self.keyword_index_path)

        self.semantic_map = load_map(self.semantic_map_path)
        self.atmosphere_map = load_map(self.atmosphere_map_path)
        self.keyword_map = load_map(self.keyword_map_path)

    def _save_faiss_indices(self):
        """
        将内存中的 FAISS 向量索引和映射表持久化保存到磁盘。

        原理：FAISS 索引数据存储在内存中，程序退出后会丢失。
        本方法将三个维度的索引分别用 faiss.write_index 序列化为二进制文件，
        并将映射表（dict）以 JSON 格式写入对应文件。
        仅在索引中有数据（ntotal > 0）时才写入 FAISS 文件，避免创建空文件。
        每次 add_memory_episode 写入新记忆后都会调用此方法，确保数据实时落盘。
        """
        if self.semantic_index.ntotal > 0: faiss.write_index(self.semantic_index, self.semantic_index_path)
        if self.atmosphere_index.ntotal > 0: faiss.write_index(self.atmosphere_index, self.atmosphere_index_path)
        if self.keyword_index.ntotal > 0: faiss.write_index(self.keyword_index, self.keyword_index_path)

        def save_map(data, path):
            """将映射表字典保存为 JSON 文件"""
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        save_map(self.semantic_map, self.semantic_map_path)
        save_map(self.atmosphere_map, self.atmosphere_map_path)
        save_map(self.keyword_map, self.keyword_map_path)

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的归一化向量嵌入（内部工具方法）。

        原理：调用 OllamaEmbedder 获取原始向量，然后进行 L2 归一化。
        归一化确保所有向量模长为 1，使得 FAISS 的 IndexFlatIP（内积检索）
        等价于余弦相似度检索。若嵌入失败（如 API 不可用），
        返回零向量以避免程序中断，但零向量在检索时不会产生有效匹配。

        :param text: 待嵌入的文本
        :return: 归一化后的 768 维 numpy 向量，失败时返回零向量
        """
        vec = self.embedder.get_embedding(text)
        if vec is None:
            return np.zeros(self.dim, dtype=np.float32)
        return normalize_vector(vec)

    def _add_to_faiss(self, index, vector: np.ndarray, mapping: dict, db_id: int) -> int:
        """
        将向量安全写入 FAISS 索引，并维护映射关系。

        原理：FAISS 的 IndexFlat 系列索引使用连续自增的内部 ID。
        本方法将向量添加到指定索引中，并在映射表中建立
        "FAISS 内部 ID → 数据库记录 ID"的对应关系。
        为处理潜在的 ID 冲突（如索引被部分重建后 ntotal 与映射表不同步），
        会检查即将使用的 faiss_id 是否已存在于映射表中，若冲突则递增直到找到空位。
        向量需 reshape 为 (1, dim) 的二维数组以满足 FAISS 的输入要求。

        :param index: FAISS 索引实例
        :param vector: 待写入的向量（1D numpy 数组）
        :param mapping: 映射表字典（faiss_id_str → db_id）
        :param db_id: SQLite 数据库中的记录 ID
        :return: 分配的 FAISS 内部 ID
        """
        faiss_id = index.ntotal
        while str(faiss_id) in mapping:
            faiss_id += 1
        index.add(vector.reshape(1, -1).astype(np.float32))
        mapping[str(faiss_id)] = db_id
        return faiss_id

    # =====================================================================
    # 核心写接口 (Write)
    # =====================================================================

    def add_memory_episode(self, episode_data: dict,precomputed_vec: Optional[np.ndarray] = None) -> int:
        """
        写入一条完整的记忆切片，同时更新 SQLite 和三个 FAISS 向量索引。

        原理：一条记忆的写入需要在两个存储系统中同步完成：
        1. SQLite 写入：将结构化数据（时间戳、叙事、对话、氛围、关键词、重要性）
           插入 memory_episodes 表，获取自增主键 db_id；
        2. 语义向量入库：对 narrative 文本生成嵌入向量（或使用预计算的向量），
           写入 semantic FAISS 索引；
        3. 氛围向量入库：对 atmosphere 文本生成嵌入向量，写入 atmosphere FAISS 索引；
        4. 关键词向量入库：对 keywords 列表中的每个关键词分别生成嵌入向量，
           逐一写入 keyword FAISS 索引（一条记忆可对应多个关键词向量）；
        5. 持久化保存所有 FAISS 索引到磁盘。

        :param episode_data: 记忆数据字典，必须包含 timestamp、narrative，
                            可选 raw_dialogue、atmosphere、keywords(list)、importance
        :param precomputed_vec: 预计算的语义向量（可选），若提供则跳过 narrative 的嵌入计算
        :return: 新记录的数据库 ID
        """
        # 1. 写入 SQLite
        cursor = self.conn.execute("""
            INSERT INTO memory_episodes 
            (timestamp, narrative, raw_dialogue, atmosphere, keywords, importance)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            episode_data["timestamp"],
            episode_data["narrative"],
            json.dumps(episode_data.get("raw_dialogue",[]), ensure_ascii=False),
            episode_data.get("atmosphere", ""),
            json.dumps(episode_data.get("keywords",[]), ensure_ascii=False),
            float(episode_data.get("importance", 0.5))
        ))
        db_id = cursor.lastrowid
        self.conn.commit()

        # 2. 语义向量入库 (Narrative)
        vec = precomputed_vec if precomputed_vec is not None else self._get_embedding(episode_data["narrative"])
        self._add_to_faiss(self.semantic_index, vec, self.semantic_map, db_id)

        # 3. 氛围向量入库 (Atmosphere)
        if episode_data.get("atmosphere"):
            vec = self._get_embedding(episode_data["atmosphere"])
            self._add_to_faiss(self.atmosphere_index, vec, self.atmosphere_map, db_id)

        # 4. 关键词向量入库 (Keywords - 一对多映射)
        keywords = episode_data.get("keywords",[])
        if isinstance(keywords, list):
            for kw in keywords:
                if str(kw).strip():
                    vec = self._get_embedding(str(kw).strip())
                    self._add_to_faiss(self.keyword_index, vec, self.keyword_map, db_id)

        self._save_faiss_indices()
        return db_id

    # =====================================================================
    # 核心读接口 (Read / Search)
    # =====================================================================

    def get_episode_by_id(self, db_id: int) -> Optional[Dict]:
        """
        根据数据库 ID 获取完整的记忆字典。

        原理：通过主键直接查询 SQLite 获取记忆记录，并将数据库行
        转换为标准化的 Python 字典格式。其中 raw_dialogue 和 keywords
        字段从 JSON 字符串反序列化为 Python 列表。
        若 ID 不存在则返回 None。此方法是 _hydrate_results 的基础，
        用于将 FAISS 检索到的 ID 水合为完整的记忆数据。

        :param db_id: 数据库记录 ID
        :return: 记忆字典（包含 id、timestamp、narrative 等全部字段），不存在返回 None
        """
        cursor = self.conn.execute("SELECT * FROM memory_episodes WHERE id = ?", (db_id,))
        row = cursor.fetchone()
        if not row: return None
        
        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "narrative": row["narrative"],
            "raw_dialogue": json.loads(row["raw_dialogue"]) if row["raw_dialogue"] else [],
            "atmosphere": row["atmosphere"],
            "keywords": json.loads(row["keywords"]) if row["keywords"] else [],
            "importance": row["importance"]
        }

    def _search_base(self, query: str, index, mapping: dict, top_k: int) -> List[Tuple[int, float]]:
        """
        底层 FAISS 向量检索逻辑，是所有搜索方法的核心引擎。

        原理：
        1. 将查询文本转换为向量（通过 _get_embedding）；
        2. 调用 FAISS 的 index.search 执行近邻检索，请求 top_k*3 个候选
           （扩大检索范围以应对关键词一对多映射导致的重复命中）；
        3. 遍历检索结果，通过映射表将 FAISS 内部 ID 转换为数据库 ID；
        4. 去重合并：同一条记忆可能被多个关键词向量命中，保留最高相似度分数；
        5. 按相似度降序排序，截取前 top_k 个结果返回。
        返回的是 (db_id, similarity_score) 的元组列表，不含完整数据。

        :param query: 查询文本
        :param index: FAISS 索引实例（语义/氛围/关键词之一）
        :param mapping: 对应的映射表（faiss_id_str → db_id）
        :param top_k: 返回的最大结果数
        :return: [(db_id, similarity_score), ...] 列表，按相似度降序排列
        """
        if index.ntotal == 0 or not query.strip():
            return[]
        
        q_vec = self._get_embedding(query)
        distances, indices = index.search(q_vec.reshape(1, -1).astype(np.float32), top_k * 3)
        
        results_dict = {}
        for i, (faiss_id, dist) in enumerate(zip(indices[0], distances[0])):
            if faiss_id == -1: continue
            db_id = mapping.get(str(faiss_id))
            if db_id is None: continue
            
            # 如果同一个 db_id 被命中多次（比如多个关键词匹配），保留最高分
            if db_id not in results_dict or dist > results_dict[db_id]:
                results_dict[db_id] = float(dist)
                
        # 排序并截取
        sorted_results = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def search_by_semantic(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        通过整体语义进行记忆检索。

        原理：将查询文本的语义向量与所有记忆的 narrative 语义向量进行相似度比较，
        返回语义最接近的记忆。适用于模糊的、基于含义的记忆搜索场景，
        如"之前关于旅行的对话"可以匹配到包含旅行相关叙事的记忆。

        :param query: 查询文本
        :param top_k: 返回结果数量，默认 5
        :return: 记忆字典列表（含 similarity 分数），按相似度降序排列
        """
        hits = self._search_base(query, self.semantic_index, self.semantic_map, top_k)
        return self._hydrate_results(hits)

    def search_by_atmosphere(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        通过氛围/情感标签进行记忆检索。

        原理：将查询文本与记忆的 atmosphere（氛围）标签向量进行匹配，
        适用于按情感状态搜索记忆的场景，如"温馨"、"紧张"、"悲伤"等
        情绪关键词可以匹配到对应氛围的记忆片段。

        :param query: 查询文本（通常是情感/氛围描述词）
        :param top_k: 返回结果数量，默认 5
        :return: 记忆字典列表（含 similarity 分数），按相似度降序排列
        """
        hits = self._search_base(query, self.atmosphere_index, self.atmosphere_map, top_k)
        return self._hydrate_results(hits)

    def search_by_keyword(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        通过关键词进行精准记忆检索。

        原理：将查询文本与记忆的各个关键词向量进行匹配。由于每条记忆的
        多个关键词被独立嵌入并存入索引，即使查询词只匹配到其中一个关键词，
        也能召回该条记忆。适用于已知具体关键词的精确搜索场景。

        :param query: 查询文本（通常是具体的关键词或短语）
        :param top_k: 返回结果数量，默认 5
        :return: 记忆字典列表（含 similarity 分数），按相似度降序排列
        """
        hits = self._search_base(query, self.keyword_index, self.keyword_map, top_k)
        return self._hydrate_results(hits)

    def hybrid_search(self, query: str, top_k: int = 5, 
                  semantic_w: float = 0.33, atmos_w: float = 0.33, keyword_w: float = 0.34,
                  importance_power: float = 0.5) -> List[Dict]:
        """
        多维度加权混合检索，融合语义、氛围、关键词三个维度的检索结果，
        并结合记忆的重要性权重进行综合排序。

        原理：这是记忆检索的核心方法，实现了多维度 RAG（检索增强生成）策略：
        1. 分维度检索：分别在语义、氛围、关键词三个 FAISS 索引中检索候选记忆，
           每个维度请求 top_k*2 个候选以扩大召回率；
        2. 加权融合：对同一条记忆在不同维度的相似度分数进行加权求和：
           combined_score = sem_score × semantic_w + atm_score × atmos_w + kw_score × keyword_w
           默认权重近似均分（0.33/0.33/0.34），可根据场景调整；
        3. 重要性调整：从数据库查询每条记忆的 importance 值（0~1），
           将融合分数乘以 importance^importance_power 进行调整。
           当 importance_power < 1 时（默认 0.5，即取平方根），
           重要性的影响被"平滑化"——高重要性的记忆得到适度提升，
           而非线性或指数级的提升，避免低重要性记忆被完全淹没；
        4. 排序并截取 top_k 个最终结果，水合为完整的记忆字典。

        :param query: 查询文本
        :param top_k: 返回结果数量
        :param semantic_w: 语义维度权重
        :param atmos_w: 氛围维度权重
        :param keyword_w: 关键词维度权重
        :param importance_power: 重要性幂次，<1 时重要性影响更平滑
        :return: 记忆字典列表（含 similarity 分数），按综合得分降序排列
        """
        # 1. 分别检索各维度
        sem_hits = self._search_base(query, self.semantic_index, self.semantic_map, top_k * 2)
        atm_hits = self._search_base(query, self.atmosphere_index, self.atmosphere_map, top_k * 2)
        kw_hits = self._search_base(query, self.keyword_index, self.keyword_map, top_k * 2)

        # 2. 合并原始得分（不包含重要性）
        combined_raw = {}
        def merge_hits(hits, weight):
            """将单个维度的检索结果按权重累加到合并字典中"""
            for db_id, score in hits:
                combined_raw[db_id] = combined_raw.get(db_id, 0.0) + (score * weight)

        merge_hits(sem_hits, semantic_w)
        merge_hits(atm_hits, atmos_w)
        merge_hits(kw_hits, keyword_w)

        if not combined_raw:
            return []

        # 3. 获取 importance 并调整得分
        combined_adjusted = []
        for db_id, raw_score in combined_raw.items():
            # 从数据库查询 importance
            cursor = self.conn.execute("SELECT importance FROM memory_episodes WHERE id = ?", (db_id,))
            row = cursor.fetchone()
            if row:
                importance = float(row[0])  # 0.0 ~ 1.0
            else:
                importance = 0.5  # 默认值

            # 调整得分：原始得分 * (importance ** importance_power)
            # 重要性越高，调整系数越大（>1 或 <1）
            # 当 importance_power < 1 时，重要性较低的分数会被压低，重要性高的分数会被提升
            adjusted_score = raw_score * (importance ** importance_power)
            combined_adjusted.append((db_id, adjusted_score))

        # 4. 排序并截取 top_k
        combined_adjusted.sort(key=lambda x: x[1], reverse=True)
        top_ids = combined_adjusted[:top_k]

        # 5. 水合结果
        return self._hydrate_results(top_ids)

    def _hydrate_results(self, hits: List[Tuple[int, float]]) -> List[Dict]:
        """
        将检索命中的 ID-分数列表"水合"为完整的记忆字典列表。

        原理："水合"（Hydrate）是数据库领域的术语，指将简单的 ID 引用
        补充为包含完整字段的数据对象。本方法遍历 FAISS 检索返回的
        (db_id, similarity_score) 列表，通过 get_episode_by_id
        从 SQLite 查询每条记忆的完整数据，并将相似度分数附加到结果中。
        若某个 ID 在数据库中不存在（如数据被删除），则跳过该条目。

        :param hits: [(db_id, similarity_score), ...] 检索结果列表
        :return: 完整记忆字典列表，每条字典额外包含 "similarity" 字段
        """
        results =[]
        for db_id, score in hits:
            ep = self.get_episode_by_id(db_id)
            if ep:
                ep["similarity"] = score
                results.append(ep)
        return results

    def stats(self) -> Dict:
        """
        获取记忆库的宏观统计信息，用于监控系统健康度。

        原理：汇总记忆库的关键指标，包括：
        - total_memory_episodes: SQLite 中的记忆总条数
        - semantic/atmosphere/keyword_vectors: 三个 FAISS 索引中的向量总数
          （关键词向量数通常 > 记忆条数，因为每条记忆可有多个关键词）
        - db_path: 数据存储路径
        这些信息用于检查数据一致性（如 SQLite 条数与 FAISS 向量数的对应关系）
        和监控存储增长情况。包含异常捕获以防止监控查询导致服务中断。

        :return: 统计信息字典，出错时包含 "error" 字段
        """
        try:
            cursor = self.conn.execute("SELECT COUNT(*) FROM memory_episodes")
            total_episodes = cursor.fetchone()[0]
            
            return {
                "total_memory_episodes": total_episodes,
                "semantic_vectors": self.semantic_index.ntotal,
                "atmosphere_vectors": self.atmosphere_index.ntotal,
                "keyword_vectors": self.keyword_index.ntotal,
                "db_path": self.db_path
            }
        except Exception as e:
            return {"error": str(e)}

    def reset(self):
        """
        物理重置：彻底清空数据库与向量索引，恢复为初始空白状态。

        原理：这是一个破坏性操作，执行以下步骤：
        1. 关闭当前 SQLite 连接；
        2. 删除所有 FAISS 索引文件（.faiss）和映射表文件（_map.json）；
        3. 删除 SQLite 数据库文件（memory_episodes.db）；
        4. 重新创建 SQLite 连接和空表；
        5. 重新初始化空的 FAISS 索引。
        该操作不可逆，所有存储的记忆将永久丢失。
        仅用于记忆库重构、数据迁移或系统彻底重启等特殊场景。
        """
        self.conn.close()
        
        # 删除所有索引文件
        for path in [self.semantic_index_path, self.atmosphere_index_path, self.keyword_index_path,
                     self.semantic_map_path, self.atmosphere_map_path, self.keyword_map_path]:
            if os.path.exists(path):
                os.remove(path)
        
        # 删除数据库文件
        db_file = os.path.join(self.db_path, "memory_episodes.db")
        if os.path.exists(db_file):
            os.remove(db_file)
            
        # 重新初始化
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()
        self._init_faiss_indices()
        self._load_faiss_indices()

    def close(self):
        """
        安全关闭记忆库，先持久化再断开连接。

        原理：关闭前先调用 _save_faiss_indices 将内存中的向量索引
        和映射表保存到磁盘，确保所有数据都已落盘，然后关闭 SQLite 连接。
        这是程序正常退出时应调用的方法，防止因进程异常终止导致数据丢失。
        """
        self._save_faiss_indices()
        self.conn.close()