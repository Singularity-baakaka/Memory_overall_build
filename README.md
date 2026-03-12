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

---

## 数学基础知识

本节面向希望快速理解项目底层原理的开发者，系统讲解项目中涉及的核心数学概念。
掌握这些知识后，你将能够自信地阅读、修改和扩展本项目的检索与向量计算逻辑。

### 1. 向量与文本嵌入（Text Embedding）

**核心思想**：将自然语言文本映射为固定维度的数值向量，使语义相似的文本在向量空间中距离更近。

本项目使用 Ollama 部署的 `gte-base-zh` 模型将文本编码为 **768 维**的浮点向量：

```
"今天天气很好" → [0.12, -0.34, 0.56, ..., 0.08]   # 768 个浮点数
"阳光明媚的一天" → [0.11, -0.32, 0.55, ..., 0.09]   # 语义相似 → 向量接近
"数据库连接失败" → [-0.45, 0.67, -0.12, ..., 0.33]  # 语义不同 → 向量远离
```

> **对应代码**：`OllamaEmbedder.get_embedding(text)` → 返回 768 维 `List[float]`

### 2. L2 归一化（L2 Normalization）

**数学定义**：将向量除以其 L2 范数（欧几里得长度），使向量的模长变为 1。

$$
\hat{v} = \frac{v}{\|v\|_2} = \frac{v}{\sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}}
$$

**直观理解**：归一化后的向量只保留了**方向信息**，丢弃了长度信息。就好像把所有向量都投射到一个单位球面上。

**为什么需要归一化？**

- 归一化后，两个向量的**点积**（内积）**恰好等于**它们的**余弦相似度**（数学证明见下文）。
- FAISS 的 `IndexFlatIP`（内积索引）在接收归一化向量后，可以直接用内积替代余弦相似度检索，计算效率更高。

```python
# 对应代码：normalize_vector(vec)
v = np.array(vec, dtype=np.float32)
norm = np.linalg.norm(v)    # 计算 ||v||₂
if norm > 0:
    return v / norm          # 单位化：模长变为 1
return v                     # 零向量保持不变，避免除零错误
```

> **注意**：零向量（模长为 0）无法归一化——它没有方向。代码中直接返回零向量以防止除零错误。

### 3. 余弦相似度（Cosine Similarity）

**数学定义**：衡量两个向量夹角的余弦值。

$$
\cos(\theta) = \frac{v_1 \cdot v_2}{\|v_1\| \times \|v_2\|}
$$

其中 $v_1 \cdot v_2 = \sum_{i=1}^{n} v_{1i} \times v_{2i}$ 是向量点积。

**值域与含义**：

| 余弦值 | 夹角 | 含义 |
|---------|------|------|
| `1` | 0° | 方向完全相同（语义最相似） |
| `0` | 90° | 完全正交（语义无关） |
| `-1` | 180° | 方向完全相反（语义相反） |

**为什么选择余弦相似度而不是欧几里得距离？**

余弦相似度只关注向量的**方向**，不受向量**长度**的影响。在文本嵌入场景中，不同长度的文本产生的向量模长可能不同，但它们的语义方向才是我们关心的。

```python
# 对应代码：cosine_similarity(v1, v2)
# 分母添加 1e-8 极小值，防止除零错误
return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
```

### 4. 内积与 FAISS IndexFlatIP 的等价关系

**关键定理**：当两个向量都经过 L2 归一化后，它们的内积等于余弦相似度。

$$
\hat{v}_1 \cdot \hat{v}_2 = \frac{v_1}{\|v_1\|} \cdot \frac{v_2}{\|v_2\|} = \frac{v_1 \cdot v_2}{\|v_1\| \times \|v_2\|} = \cos(\theta)
$$

**在本项目中的应用**：

1. 所有向量在存入 FAISS 之前都经过 `normalize_vector()` 归一化；
2. FAISS 使用 `IndexFlatIP`（**I**nner **P**roduct，内积索引），检索时计算查询向量与所有库中向量的内积；
3. 因为归一化的保证，内积结果 = 余弦相似度 → 分数越高 = 语义越相似。

这种设计避免了每次检索都要做除法归一化，让 FAISS 能够使用更高效的内积计算来完成余弦相似度检索。

> **对应代码**：`MemoryDB._get_embedding()` 调用 `normalize_vector()` 后存入 `IndexFlatIP`

### 5. 加权平均与稳健嵌入（Robust Embedding）

直接对整段文本做嵌入可能导致局部细节丢失。`compute_robust_embedding` 采用**分而治之 + 加权平均**的策略：

#### 第一步：分句

```
"今天和小明去了公园。我们玩得很开心。下次还要一起去！"
        ↓ split_sentences()
["今天和小明去了公园", "我们玩得很开心", "下次还要一起去"]
```

#### 第二步：逐句嵌入 + 权重计算

每个句子的权重由**位置权重**和**长度权重**的乘积决定：

$$
w_i = w_{\text{pos}}(i) \times w_{\text{len}}(i)
$$

- **位置权重** $w_{\text{pos}}$：首句和尾句为 `1.2`，中间句为 `1.0`。依据认知科学中的**首因效应**（Primacy Effect）和**近因效应**（Recency Effect）——人们倾向于更好地记住开头和结尾的信息。
  
- **长度权重** $w_{\text{len}}$：$\min\left(\frac{\text{字符数}}{20},\ 1.5\right)$。较长的句子通常信息密度更高，但设置上限 1.5 避免超长句子过度主导。

#### 第三步：加权平均 + 归一化

$$
\vec{v}_{\text{robust}} = \text{normalize}\left(\frac{\sum_{i} w_i \cdot \vec{v}_i}{\sum_{i} w_i}\right)
$$

> **对应代码**：`compute_robust_embedding(narrative, embedder)`

### 6. 多维度加权混合检索（Hybrid Search）

`hybrid_search` 是本项目最核心的检索算法。它从三个维度同时检索记忆，并融合得分。

#### 第一步：三维度并行检索

| 维度 | 索引 | 匹配目标 | 适用场景 |
|------|------|----------|---------|
| 语义（Semantic） | `semantic_index` | narrative 叙事内容 | "关于旅行的记忆" |
| 氛围（Atmosphere） | `atmosphere_index` | atmosphere 情感标签 | "温馨的回忆" |
| 关键词（Keyword） | `keyword_index` | keywords 实体列表 | "提到小明的记忆" |

#### 第二步：加权融合

对于被命中的每条记忆，将其在各维度的相似度分数按权重加权求和：

$$
S_{\text{raw}} = S_{\text{sem}} \times w_{\text{sem}} + S_{\text{atm}} \times w_{\text{atm}} + S_{\text{kw}} \times w_{\text{kw}}
$$

默认权重近似均分：$w_{\text{sem}} = 0.33$，$w_{\text{atm}} = 0.33$，$w_{\text{kw}} = 0.34$。

> **调参建议**：如果你的应用更关注精确实体匹配（如角色扮演中的人名），可以调高 `keyword_w`；如果更关注情绪共鸣，可以调高 `atmos_w`。

#### 第三步：重要性幂函数调整

$$
S_{\text{final}} = S_{\text{raw}} \times \text{importance}^{p}
$$

其中 `importance` ∈ [0, 1] 是记忆的重要性权重，`p`（默认 0.5）是 `importance_power` 参数。

**为什么用幂函数（power function）？**

当 `p = 0.5` 时，即取平方根 $\sqrt{\text{importance}}$，效果如下：

| importance | importance^1.0（线性） | importance^0.5（平方根） |
|------------|----------------------|------------------------|
| 1.0 | 1.0 | 1.0 |
| 0.8 | 0.8 | 0.894 |
| 0.5 | 0.5 | 0.707 |
| 0.2 | 0.2 | 0.447 |
| 0.1 | 0.1 | 0.316 |

**直觉理解**：平方根比线性更加"平滑"——低重要性的记忆不会被过度压低，仍有机会被召回。这避免了重要性标注偏差导致某些记忆被完全埋没的问题。

- 当 `p = 0` 时，$\text{importance}^0 = 1$，重要性不影响排序；
- 当 `p = 1` 时，重要性线性影响得分；
- 当 `p > 1` 时，重要性差异被放大（更激进地优先重要记忆）。

> **对应代码**：`MemoryDB.hybrid_search()`

### 概念关系总图

```
                        ┌─────────────────────────────────┐
                        │          原始文本输入             │
                        └────────────┬────────────────────┘
                                     │
                        ┌────────────▼────────────────────┐
                        │  OllamaEmbedder.get_embedding() │
                        │    文本 → 768 维原始向量          │
                        └────────────┬────────────────────┘
                                     │
                        ┌────────────▼────────────────────┐
                        │     normalize_vector()           │
                        │   L2 归一化 → 单位向量            │
                        └────────────┬────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
   ┌──────────▼──────┐   ┌──────────▼──────┐   ┌──────────▼──────┐
   │  语义索引 (IP)   │    │  氛围索引 (IP)   │   │ 关键词索引 (IP)   │
   │  IndexFlatIP    │   │  IndexFlatIP    │   │  IndexFlatIP    │
   └──────────┬──────┘   └──────────┬──────┘   └──────────┬──────┘
              │                      │                      │
              │   内积 = 余弦相似度     │    内积 = 余弦相似度    │
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     │
                        ┌────────────▼────────────────────┐
                        │   hybrid_search() 加权融合        │
                        │ S = Σ(score_i × weight_i)        │
                        │ S_final = S × importance^p       │
                        └─────────────────────────────────┘
```

---

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

---

#### D. `txt_to_vectordb.py` - 对话文本固化脚本

本模块是整个项目的**入口管道**，负责把一份普通的 txt 聊天记录，经过解析→切片→摘要→入库，变成可被语义检索的向量记忆库。

> **整体数据流**
> ```
> txt 文件
>   ↓ parse_dialogue_file()       逐行识别角色和内容
> 对话列表 [{role, content}, ...]
>   ↓ chunk_dialogues()           按固定轮数切成若干片段
> 对话片段 [[...], [...], ...]
>   ↓ generate_episode_summary()  (LLM模式) 调 LLM 生成摘要/氛围/关键词
>   ↓ build_episode_data_fast()   (快速模式) 直接用原文+正则提取关键词
> episode_data 字典
>   ↓ MemoryDB.add_memory_episode()
> SQLite + FAISS 向量记忆库（持久化，可检索）
> ```

**1. `parse_dialogue_file(filepath)`**

*   **功能**：读取 txt 对话文件，逐行解析成结构化的对话列表。
*   **原理**：
    用正则 `r'^(.+?)[：:]\s*(.*)'` 匹配每一行——同时支持中英文冒号：
    - 匹配成功 → 新建一条 `{"role": 角色名, "content": 内容}` 字典；
    - 匹配失败（无角色前缀）→ 视为上一条消息的续行，直接追加到 `content` 末尾；
    - 空行跳过，不影响解析。
*   **效果**：
    ```
    "用户: 你好\n今天天气不错\n林梓墨: 是啊！"
                    ↓
    [{"role": "用户", "content": "你好\n今天天气不错"},
     {"role": "林梓墨", "content": "是啊！"}]
    ```
    文件不存在时抛出 `FileNotFoundError`，空文件返回 `[]`。

**2. `chunk_dialogues(dialogues, chunk_size=20)`**

*   **功能**：将完整对话列表按固定轮数切成多个片段，每个片段将成为独立的一条记忆。
*   **原理**：
    用 Python 切片 `dialogues[i : i+chunk_size]`，步长为 `chunk_size` 滑动截取。
    内置最小值保护：`chunk_size < 2` 时强制设为 2（单条消息不构成有意义对话）。
*   **效果**：

    | chunk_size | 12条对话 → 片段数 | 特点 |
    |---|---|---|
    | 4 | 3 片（4+4+4） | 粒度细，检索精度高 |
    | 20 | 1 片（12条全在） | 粒度粗，上下文更完整 |

**3. `_SUMMARY_PROMPT`（模块级常量）**

*   **功能**：发送给 LLM 的提示词模板，约束输出严格 JSON 格式。
*   **原理**：
    提示词要求 LLM 扮演"记忆整理专家"，输出三个字段：
    `narrative`（100字内第三人称叙事摘要）、`atmosphere`（2-4个情感词）、`keywords`（关键词列表）。
    只输出 JSON，禁止任何额外文字，减少 `safe_parse_json` 的清洗压力。

**4. `generate_episode_summary(dialogue_chunk)`**

*   **功能**：调用 LLM，为一段对话生成高质量叙事摘要、氛围标签和关键词（LLM 模式专用）。
*   **原理**：
    1. 把对话列表拼成 `"角色: 内容"` 多行文本；
    2. 构造 `messages`（system + user）发送给 LLM，`temperature=0.3` 降低随机性；
    3. `safe_parse_json()` 解析返回结果（自动清除 LLM 可能包裹的 ` ```json ``` ` 标记）；
    4. 提取 `narrative / atmosphere / keywords` 三个字段返回。
*   **效果**：输入10条对话 → 输出 `{"narrative": "...", "atmosphere": "好奇, 轻松", "keywords": ["霍金", "黑洞"]}` 。若 LLM 失败或格式不合法，返回 `None`，上层自动降级到快速模式。

**5. `build_episode_data(dialogue_chunk, summary, timestamp=None)`**

*   **功能**：将对话片段 + LLM 摘要组装为 `MemoryDB` 可直接写入的 `episode_data` 字典（纯数据组装，无业务逻辑）。
*   **原理**：从 `dialogue_chunk` 提取原始对话作为 `raw_dialogue`，从 `summary` 取摘要字段，`timestamp` 默认取当前时间，`importance` 固定 0.5。
*   **效果**：输出的字典满足 `MemoryDB.add_memory_episode()` 的接口规范，可直接入库。

**6. `build_episode_data_fast(dialogue_chunk, timestamp=None)`**

*   **功能**：不依赖 LLM，直接用对话原文构建 `episode_data`（快速/离线模式）。
*   **原理**：
    - `narrative`：直接拼接对话原文多行文本；
    - `atmosphere`：留空（无法自动推断情感）；
    - `keywords`：用正则 `r'[\u4e00-\u9fff]{2,6}'` 扫描长度 2~6 字的中文词组，去重取前 5 个。
*   **效果**：

    | 维度 | LLM 模式 | 快速模式 |
    |---|---|---|
    | narrative 质量 | 高（精炼摘要） | 低（原文堆砌） |
    | atmosphere | 有（情感标签） | 空 |
    | keywords 质量 | 高（语义提炼） | 中（正则粗提取） |
    | 速度 | 慢（需联网） | 极快（纯本地） |

**7. `txt_to_vectordb(txt_path, output_dir, chunk_size, use_llm, embedder)`**

*   **功能**：整个模块的主入口函数，一键将 txt 文件固化为可检索的向量记忆库。
*   **原理**：串联所有步骤，包含 LLM 失败时的自动降级容错。执行完毕后 `output_dir` 中生成 `.db`、`.faiss`、`_map.json` 等持久化文件。
*   **效果**：返回统计字典 `{status, total_dialogues, total_chunks, success, failed, db_stats}`。

**8. `main()`**

*   **功能**：命令行入口，`argparse` 解析参数后调用 `txt_to_vectordb()`。
*   **效果**：支持直接从终端运行，例如：
    ```bash
    python txt_to_vectordb.py chat.txt --output ./my_db --chunk-size 10 --fast
    ```

---

#### E. `test_txt_to_vectordb.py` - 单元与集成测试

本模块通过 `unittest` 框架对 `txt_to_vectordb.py` 中的所有核心函数进行测试，**无需启动任何外部服务**（`MockEmbedder` 替代真实 Ollama）。

**`MockEmbedder` 类**

*   **功能**：模拟 `OllamaEmbedder`，在无 Ollama 服务时提供确定性伪向量。
*   **原理**：
    1. `hash(text) % 2^31` 得到非负随机种子；
    2. `np.random.seed(种子)` 初始化，保证**相同文本始终产生相同向量**；
    3. 标准正态分布采样 768 维，L2 归一化后返回，接口与真实嵌入器完全兼容。
*   **效果**：测试无需网络，任何机器稳定复现，FAISS 仍能产生差异化相似度排序。

**测试数据常量**

| 常量名 | 内容 | 用途 |
|---|---|---|
| `SAMPLE_DIALOGUE` | 12条对话（书籍+电影两个话题） | 解析数量验证、端到端检索验证 |
| `SAMPLE_DIALOGUE_ENGLISH_COLON` | 2条，使用英文冒号 | 冒号兼容性测试 |
| `SAMPLE_MULTILINE` | 3条，含续行 | 多行拼接逻辑测试 |

**`TestParseDialogueFile`** — 测试 `parse_dialogue_file()`

| 测试方法 | 验证内容 |
|---|---|
| `test_basic_parse` | 中文冒号格式能正确识别角色和内容 |
| `test_english_colon` | 英文冒号 `:` 也能被正确解析 |
| `test_multiline_content` | 续行被正确拼接到上一条消息的 content 中 |
| `test_empty_file` | 空文件返回 `[]`，不抛异常 |
| `test_file_not_found` | 文件不存在时抛出 `FileNotFoundError` |
| `test_dialogue_count` | SAMPLE_DIALOGUE 恰好解析出 12 条（回归测试） |

**`TestChunkDialogues`** — 测试 `chunk_dialogues()`

| 测试方法 | 验证内容 |
|---|---|
| `test_basic_chunking` | 10条/chunk=4 → 3片（4+4+2），尾部不足也保留 |
| `test_exact_division` | 8条/chunk=4 → 恰好 2片，无多余空片 |
| `test_single_chunk` | chunk_size 大于总数时，全部在 1 个片段中 |
| `test_empty_dialogues` | 空输入返回 `[]` |
| `test_min_chunk_size` | `chunk_size=1` 被强制升为 2 |

**`TestBuildEpisodeDataFast`** — 测试 `build_episode_data_fast()`

| 测试方法 | 验证内容 |
|---|---|
| `test_fast_build` | 输出包含全部必要字段，narrative 含原文，raw_dialogue 数量正确 |
| `test_fast_build_keywords` | 从中文对话成功正则提取到非空关键词列表 |

**`TestBuildEpisodeData`** — 测试 `build_episode_data()`（不调真实 LLM）

| 测试方法 | 验证内容 |
|---|---|
| `test_build_with_summary` | narrative/atmosphere/keywords 正确从模拟 summary 映射，raw_dialogue 数量正确 |

**`TestFullPipeline`** — 端到端集成测试 `txt_to_vectordb()`

*   **测试策略**：`MockEmbedder` + `tempfile.mkdtemp()` 临时沙箱，`tearDown` 自动清理，各测试相互隔离。

| 测试方法 | 验证内容 |
|---|---|
| `test_full_pipeline_fast_mode` | 统计字典正确（total=12, failed=0）；重新打开库后语义检索"电影"有命中，结果包含 narrative/similarity/raw_dialogue |
| `test_pipeline_creates_db_files` | 固化后 `memory_episodes.db` 和 `semantic.faiss` 文件确实落盘 |
| `test_pipeline_different_chunk_sizes` | chunk_size=4 产生的片段数严格大于 chunk_size=20 产生的片段数 |

