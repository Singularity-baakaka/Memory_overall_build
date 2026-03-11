# Memory_overall_build
提供方法，便捷快速的把AI对话文本变换成向量记忆库，提供更强的RAG基底

## 核心模块与功能详解

本项目旨在构建一个多维度的向量记忆数据库，通过混合检索机制增强 AI 的 RAG（检索增强生成）能力。以下是各模块的详细原理解析：

### 1. `memory_db.py`: 记忆数据库核心 (MemoryDB)

这是系统的核心类，负责管理记忆的“写入”与“召回”。它不包含复杂的业务逻辑，而是专注于数据的持久化和高效检索。

*   **原理 (Principle)**
    采用 **混合存储架构 (Hybrid Storage)**：
    *   **结构化存储 (SQLite)**: 使用 SQLite 数据库存储记忆的完整细节，包括时间戳、叙事内容、原始对话、氛围标签、关键词列表和重要性权重。这保证了数据的完整性和可读性。
    *   **向量索引 (FAISS)**: 使用 FAISS 库维护三个独立的向量空间：
        1.  **语义索引 (Semantic)**: 基于叙事内容 (`narrative`)，用于捕捉“发生了什么”。
        2.  **氛围索引 (Atmosphere)**: 基于氛围描述 (`atmosphere`)，用于捕捉“当时的情绪/基调”。
        3.  **关键词索引 (Keyword)**: 基于提取的实体 (`keywords`)，用于精确匹配特定的物体、人名或概念。

*   **实现的效果 (Effect)**
    这种设计解决了传统单一向量检索的局限性。AI 不仅能根据“意思”找回记忆，还能根据“情绪”或“特定关键词”进行复合检索。例如，系统可以优先召回“悲伤的（氛围）”且“很重要（重要性权重）”的关于“用户童年（语义）”的记忆。

*   **主要函数功能 (Functions)**
    *   `__init__`: 初始化数据库连接和 FAISS 索引。支持数据的持久化加载。
    *   `add_memory_episode(episode_data)`: **核心写入接口**。
        *   接收包含 `narrative` (叙事), `atmosphere` (氛围), `keywords` (关键词) 等字段的字典。
        *   将元数据存入 SQLite。
        *   调用 Embedder 生成向量，分别存入三个 FAISS 索引。
        *   自动处理 ID 映射，确保向量索引与数据库记录一一对应。
    *   `search_by_semantic` / `search_by_atmosphere` / `search_by_keyword`: **单维度检索**。分别在对应的向量空间中查找最相似的记忆。
    *   `hybrid_search(query, ...)`: **加权混合检索**。
        *   这是一个高级检索算法。它同时在三个向量空间中搜索。
        *   根据传入的权重参数 (`semantic_w`, `atmos_w`, `keyword_w`) 合并得分。
        *   **重要性加权**: 还会结合记忆的 `importance` 属性调整最终得分，使“重要”的记忆更容易浮现。
    *   `get_episode_by_id(db_id)`: 根据数据库 ID 还原完整的记忆详情。

### 2. `utils.py`: 基础设施与工具

提供底层的向量化能力和数学运算支持。

*   **原理 (Principle)**
    封装了外部模型接口（Ollama）和基础数学运算（NumPy），为上层业务提供纯净的工具支持。此外，还包含了文件 IO、JSON 处理以及 LLM API 调用的封装。

*   **主要功能 (Functions)**

    *   **向量与数学工具**
        *   `class OllamaEmbedder`: **向量嵌入器**。
            *   连接本地运行的 Ollama 服务（默认模型 `gte-base-zh`）。
            *   将文本转化为 768 维度的浮点向量。
            *   包含重试机制，提高稳定性。
        *   `normalize_vector(vec)`: **向量归一化**。将向量模长转换为 1，这是计算余弦相似度的前提，能提高后续计算效率。
        *   `cosine_similarity(v1, v2)`: 计算两个向量的余弦相似度，用于评估文本之间的语义距离。
        *   `vector_to_blob(vector)` / `blob_to_vector(blob)`: 处理 NumPy 数组与 SQLite BLOB 二进制格式之间的转换，用于数据库存储。
        *   `split_sentences(text)`: **智能分句**。
            *   根据中英文标点（。！？...）将长文本切分为完整的句子列表。
            *   保持语义完整性，为后续的精细化处理做准备。
        *   `compute_robust_embedding(narrative, embedder)`: **稳健语义向量计算**。
            *   不直接对整段文本做 Embedding，而是先分句。
            *   对每个句子分别计算向量。
            *   **加权融合**: 对开头和结尾的句子给予更高权重（位置权重），并考虑句子长度（信息密度）。
            *   这种方法生成的向量比直接 Embedding 整段文本更能代表核心语义。

    *   **文件与数据处理**
        *   `ensure_dir(path)`: 确保目录存在，如果不存在则自动创建。
        *   `read_file_content(filepath)` / `overwrite_file` / `append_to_file`: 封装的安全文件读写操作，包含错误处理。
        *   `safe_parse_json(json_str)`: **鲁棒的 JSON 解析**。能自动去除 Markdown 代码块标记（```json ... ```），防止因 LLM 返回格式不规范导致的解析错误。
        *   `extract_tag_content(text, tag_name)` / `extract_dialogue_from_stream`: XML/Tag 解析工具，用于从 LLM 的结构化输出中提取特定内容。

    *   **LLM 交互**
        *   `call_chat_api(...)`: **LLM API 调用封装**。
            *   统一封装了对 DeepSeek 等兼容 OpenAI 格式接口的调用。
            *   自动读取 `config.py` 中的默认配置。
            *   支持重试机制、超时控制和 JSON 模式。

### 3. `config.py`: 全局配置

*   `get_llm_config()`
    *   **功能**: 返回一个包含 LLM API 配置的字典。
    *   **原理**: 读取硬编码在文件中的全局变量 `LLM_CONFIG`。
    *   **效果**: 为项目中的 LLM 调用提供统一的配置入口，通过修改此处即可全局调整 API Key、模型 URL、以及 `temperature` 等生成参数。

### 详细函数列表与注释

为了更清晰地展示库的功能，以下是对项目中每个核心函数的详细讲解。

#### A. `memory_db.py` - 记忆库 DAO

1.  `__init__(self, db_path, embedder)`
    *   **功能**: 初始化 MemoryDB 实例。
    *   **原理**: 创建或连接 SQLite 数据库，创建数据表结构；初始化 FAISS 语义、氛围、关键词三个向量索引；加载已保存的 FAISS 索引文件。
    *   **效果**: 启动记忆库服务，准备好读写环境。如果本地有数据，会自动加载。

2.  `_init_tables(self)`
    *   **功能**: 初始化 SQLite 数据库表。
    *   **原理**: 执行 SQL `CREATE TABLE IF NOT EXISTS` 语句。
    *   **效果**: 确保数据库中存在 `memory_episodes` 表，包含 id, timestamp, narrative, atmosphere 等字段。

3.  `_init_faiss_indices(self)` & `_load_faiss_indices(self)`
    *   **功能**: 初始化和加载 FAISS 索引及其 ID 映射。
    *   **原理**: 检查本地是否有 `.faiss` 文件，有则读取，无则创建新的 `IndexFlatIP` (内积索引)。同时加载 json 格式的 ID 映射表。
    *   **效果**: 将磁盘上的向量索引加载到内存中，准备进行快速检索。

4.  `_save_faiss_indices(self)`
    *   **功能**: 持久化保存 FAISS 索引。
    *   **原理**: 调用 `faiss.write_index` 将内存索引写入磁盘，同时将 ID 映射表写入 JSON 文件。
    *   **效果**: 防止程序关闭后向量数据丢失，确保数据持久性。

5.  `_get_embedding(self, text)`
    *   **功能**: 获取文本的向量表示。
    *   **原理**: 调用传入的 `embedder` 工具，并进行 L2 归一化。
    *   **效果**: 将文本转换为机器可计算相似度的向量。

6.  `_add_to_faiss(self, index, vector, mapping, db_id)`
    *   **功能**: 将向量加入 FAISS 索引，并记录 ID 映射。
    *   **原理**: 维护一个自增的 `faiss_id`，避免 ID 冲突，建立 `faiss_id` 到 SQLite `db_id` 的映射关系。
    *   **效果**: 确保向量库中的每一条向量都能准确对应到数据库中的某一条目。

7.  `add_memory_episode(self, episode_data, precomputed_vec=None)`
    *   **功能**: 添加一条完整的记忆切片。
    *   **原理**: 
        1. 将完整数据插入 SQLite。
        2. 计算 `narrative` (叙事) 向量存入语义索引。
        3. 计算 `atmosphere` (氛围) 向量存入氛围索引。
        4. 遍历 `keywords` (关键词) 列表，计算每个关键词的向量存入关键词索引。
        5. 保存索引。
    *   **效果**: 完成一次完整的记忆“入脑”过程，使其变得可被检索。

8.  `get_episode_by_id(self, db_id)`
    *   **功能**: 根据 ID 获取记忆详情。
    *   **原理**: 执行 SQL `SELECT *` 查询。
    *   **效果**: 将数据库中的原始行数据还原为 Python 字典格式。

9.  `_search_base(self, query, index, mapping, top_k)`
    *   **功能**: 基础向量检索函数。
    *   **原理**: 将 query 向量化，调用 FAISS 的 `search` 接口，然后通过 mapping 将返回的 `faiss_id` 转换回 `db_id`。
    *   **效果**: 返回与查询最相似的数据库 ID 列表及其相似度得分。

10. `search_by_semantic(self, query, top_k)` / `search_by_atmosphere(...)` / `search_by_keyword(...)`
    *   **功能**: 分别针对语义、氛围、关键词维度的检索封装。
    *   **原理**: 指定对应的索引和映射表调用 `_search_base`。
    *   **效果**: 提供针对性的单一维度检索能力。

11. `hybrid_search(self, query, top_k, ...)`
    *   **功能**: 混合加权检索（核心功能）。
    *   **原理**: 
        1. 同时进行语义、氛围、关键词检索。
        2. 将三者的得分按权重 (`semantic_w`, `atmos_w`...) 加权求和。
        3. 如果数据库中有 `importance` 字段，还会乘以重要性系数 (`importance ** importance_power`)。
    *   **效果**: 模拟人类回忆机制，既考虑内容的相似度，也考虑情绪的共鸣和记忆的重要性。

#### B. `utils.py` - 工具库

**OllamaEmbedder 类**
1.  `OllamaEmbedder.__init__(self, model_name, ...)`: 初始化 Embedder，设定 API 地址和模型。
2.  `OllamaEmbedder.get_embedding(self, text)`: 
    *   呼叫本地 Ollama API，获取文本的向量数据。自带重试机制。

**数学运算**
3.  `normalize_vector(vec)`: 对向量进行 L2 归一化，使得向量模长为 1，方便计算余弦相似度。
4.  `cosine_similarity(v1, v2)`: 计算两个向量夹角的余弦值，值越接近 1 表示越相似。

**向量处理**
5.  `split_sentences(text)`:
    *   **原理**: 正则匹配 `。！？` 等结束符。
    *   **效果**: 将长文本切分为句子列表，用于精细化处理。
6.  `compute_robust_embedding(narrative, embedder)`:
    *   **原理**: 分句 -> 分别向量化 -> 根据位置和长度加权 -> 平均。
    *   **效果**: 生成比直接 Embed 整段话更精准、抗噪能力更强的语义向量。

**文件与格式处理**
7.  `vector_to_blob(vector)` / `blob_to_vector(blob)`: Numpy 数组与二进制 Bytes 互转，用于 SQLite 存储。
8.  `ensure_dir(path)` / `ensure_file_exists(...)`: 文件系统辅助，确保读写不报错。
9.  `safe_parse_json(json_str)`: 尝试解析 JSON，如果失败（如包含 Markdown 符号）会自动清洗字符串后重试。
10. `extract_tag_content(text, tag_name)` / `extract_dialogue_from_stream`: 正则工具，用于从 LLM 的输出中提取 `<user>...` 或 XML 标签内容。

**LLM API**
11. `call_chat_api(messages, ...)`:
    *   **功能**: 发送对话请求到 LLM。
    *   **原理**: 封装 `requests.post`，支持 JSON 模式、自定义参数和错误重试。
    *   **效果**: 统一的 LLM 调用接口，简化业务代码。

#### C. `config.py` - 配置

1.  `get_llm_config()`: 返回全局配置字典，包含 API Key, URL, Temperature 等模型参数。
