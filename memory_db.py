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
    def __init__(self, db_path: str = "./data/memory_db", embedder: Optional[OllamaEmbedder] = None):
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
        """初始化单一主表，存储所有的记忆切片"""
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
        """三大向量索引：语义、氛围、关键词"""
        self.dim = VECTOR_DIM
        
        # 文件路径
        self.semantic_index_path = os.path.join(self.db_path, "semantic.faiss")
        self.atmosphere_index_path = os.path.join(self.db_path, "atmosphere.faiss")
        self.keyword_index_path = os.path.join(self.db_path, "keyword.faiss")
        
        self.semantic_map_path = os.path.join(self.db_path, "semantic_map.json")
        self.atmosphere_map_path = os.path.join(self.db_path, "atmosphere_map.json")
        self.keyword_map_path = os.path.join(self.db_path, "keyword_map.json")

    def _load_faiss_indices(self):
        """加载 FAISS 及其映射表 (faiss_id -> db_id)"""
        def load_index(path):
            return faiss.read_index(path) if os.path.exists(path) else faiss.IndexFlatIP(self.dim)

        def load_map(path):
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
        """持久化保存向量库"""
        if self.semantic_index.ntotal > 0: faiss.write_index(self.semantic_index, self.semantic_index_path)
        if self.atmosphere_index.ntotal > 0: faiss.write_index(self.atmosphere_index, self.atmosphere_index_path)
        if self.keyword_index.ntotal > 0: faiss.write_index(self.keyword_index, self.keyword_index_path)

        def save_map(data, path):
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        save_map(self.semantic_map, self.semantic_map_path)
        save_map(self.atmosphere_map, self.atmosphere_map_path)
        save_map(self.keyword_map, self.keyword_map_path)

    def _get_embedding(self, text: str) -> np.ndarray:
        vec = self.embedder.get_embedding(text)
        if vec is None:
            return np.zeros(self.dim, dtype=np.float32)
        return normalize_vector(vec)

    def _add_to_faiss(self, index, vector: np.ndarray, mapping: dict, db_id: int) -> int:
        """安全的 FAISS 写入方法，自动处理 ID 冲突"""
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
        无情写入记忆切片
        :param episode_data: 必须包含 timestamp, narrative, atmosphere, keywords(list), importance
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
        """根据 ID 获取完整记忆字典"""
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
        """底层 FAISS 检索逻辑"""
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
        """通过整体描述进行检索"""
        hits = self._search_base(query, self.semantic_index, self.semantic_map, top_k)
        return self._hydrate_results(hits)

    def search_by_atmosphere(self, query: str, top_k: int = 5) -> List[Dict]:
        """通过氛围标签进行检索"""
        hits = self._search_base(query, self.atmosphere_index, self.atmosphere_map, top_k)
        return self._hydrate_results(hits)

    def search_by_keyword(self, query: str, top_k: int = 5) -> List[Dict]:
        """通过关键词进行精准检索"""
        hits = self._search_base(query, self.keyword_index, self.keyword_map, top_k)
        return self._hydrate_results(hits)

    def hybrid_search(self, query: str, top_k: int = 5, 
                  semantic_w: float = 0.33, atmos_w: float = 0.33, keyword_w: float = 0.34,
                  importance_power: float = 0.5) -> List[Dict]:
        """
        基础加权混合检索，增加重要性权重。
        :param query: 查询文本
        :param top_k: 返回结果数量
        :param semantic_w: 语义权重
        :param atmos_w: 氛围权重
        :param keyword_w: 关键词权重
        :param importance_power: 重要性幂次，<1 时重要性影响更平滑
        :return: 记忆列表，每条包含相似度得分（已加权）
        """
        # 1. 分别检索各维度
        sem_hits = self._search_base(query, self.semantic_index, self.semantic_map, top_k * 2)
        atm_hits = self._search_base(query, self.atmosphere_index, self.atmosphere_map, top_k * 2)
        kw_hits = self._search_base(query, self.keyword_index, self.keyword_map, top_k * 2)

        # 2. 合并原始得分（不包含重要性）
        combined_raw = {}
        def merge_hits(hits, weight):
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
        """将检索到的 ID 列表水合为完整的字典数据"""
        results =[]
        for db_id, score in hits:
            ep = self.get_episode_by_id(db_id)
            if ep:
                ep["similarity"] = score
                results.append(ep)
        return results

    def stats(self) -> Dict:
        """获取记忆库的宏观统计信息，用于监控系统健康度"""
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
        物理重置：彻底清空数据库与向量索引。
        这是危险指令，用于记忆库重构或彻底重启。
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
        """安全关闭"""
        self._save_faiss_indices()
        self.conn.close()