# Mini-SGLang 源码深度解析 & LLM 推理框架面试准备

> 面向 LLM & Agent 应用 / 后训练方向算法实习面试，基于 Mini-SGLang (~5000行) 和 Nano-vLLM (~1200行) 两个轻量推理框架的对照学习。

---

## 一、项目概览与定位

### 1.1 两个项目是什么

| 维度 | Mini-SGLang | Nano-vLLM |
|------|-------------|-----------|
| 上游项目 | [SGLang](https://github.com/sgl-project/sglang) (LMSYS) | [vLLM](https://github.com/vllm-project/vllm) (UC Berkeley) |
| 代码量 | ~5000 行 Python | ~1200 行 Python |
| 定位 | 教学 + 可用的高性能推理框架 | 极简教学用推理框架 |
| 核心特性 | Radix Cache, Chunked Prefill, Overlap Scheduling, TP, CUDA Graph, FlashInfer/FA3/FA4 | Prefix Caching, TP, CUDA Graph, torch.compile |
| 支持模型 | Llama, Qwen2.5, Qwen3, Qwen3-MoE, Mistral | Qwen3 |
| 在线服务 | 完整 OpenAI 兼容 API + Shell | 仅离线 Python API |

### 1.2 为什么面试要理解推理框架

**面试官视角**：即使你面的是算法岗，理解推理框架说明你：
1. 知道模型实际部署时的瓶颈在哪 → 能更好地设计算法方案
2. 理解 KV Cache、Batching、Attention 优化的底层原理 → 后训练中 reward model / DPO 训练的 infra 瓶颈
3. 有系统思维 → Agent 场景中，推理吞吐直接决定 Agent 的响应延迟和并发能力
4. 读过工程代码 → 证明你不仅能调 API，还能 debug 到 infra 层

---

## 二、系统架构深度解析

### 2.1 Mini-SGLang 架构

```
┌─────────────────────────────────────────────────────────────┐
│                       API Server (FastAPI)                   │
│                  /v1/chat/completions, /generate             │
└──────────┬──────────────────────────────────┬───────────────┘
           │ ZMQ                              │ ZMQ
    ┌──────▼──────┐                    ┌──────▼──────┐
    │  Tokenizer  │                    │ Detokenizer │
    │   Worker    │                    │   Worker    │
    └──────┬──────┘                    └──────▲──────┘
           │ ZMQ                              │ ZMQ
    ┌──────▼──────────────────────────────────┴───────┐
    │              Scheduler (Rank 0)                  │
    │  ┌──────────┐ ┌───────────┐ ┌────────────────┐  │
    │  │ Prefill  │ │  Decode   │ │  Cache Manager │  │
    │  │ Manager  │ │  Manager  │ │  (Radix/Naive) │  │
    │  └──────────┘ └───────────┘ └────────────────┘  │
    │  ┌──────────┐ ┌───────────┐ ┌────────────────┐  │
    │  │  Table   │ │  Engine   │ │  Graph Runner  │  │
    │  │ Manager  │ │ (Model+   │ │  (CUDA Graph)  │  │
    │  │          │ │  KVCache) │ │                │  │
    │  └──────────┘ └───────────┘ └────────────────┘  │
    └────────────────────┬─────────────────────────────┘
                         │ NCCL / PyNCCL
    ┌────────────────────▼─────────────────────────────┐
    │              Scheduler (Rank 1..N-1)              │
    │              (same structure as Rank 0)            │
    └──────────────────────────────────────────────────┘
```

**关键设计点**：
- **多进程架构**：API Server、Tokenizer、Detokenizer、每个 TP Rank 的 Scheduler 都是独立进程
- **通信方式**：控制消息走 ZMQ（轻量 IPC），大 Tensor 走 NCCL（GPU 间高速通信）
- **全局 Context 模式**：`Context` 对象通过全局变量传递，避免层层传参（`core.py:101-136`）

### 2.2 Nano-vLLM 架构

```
┌─────────────────────────────┐
│         LLM Engine          │
│  ┌───────┐  ┌────────────┐  │
│  │Scheduler│ │ ModelRunner │ │
│  │(waiting │ │ (model +   │ │
│  │/running)│ │  KVCache)  │ │
│  └───────┘  └────────────┘  │
└─────────────────────────────┘
        │ NCCL (TP)
┌───────▼──────────────────┐
│  ModelRunner (Rank 1..N) │
│  (via SharedMemory IPC)  │
└──────────────────────────┘
```

**关键区别**：Nano-vLLM 是单机多进程，通过 `SharedMemory` + `pickle` 做 IPC（简单但不灵活）；Mini-SGLang 用 ZMQ（可跨机、更解耦）。

---

## 三、核心概念深度剖析（面试高频考点）

### 3.1 KV Cache 与 PagedAttention

#### 3.1.1 为什么需要 KV Cache

自回归生成中，每生成一个 token 都需要 attend 到之前所有 token 的 Key 和 Value。如果每次都重新计算，复杂度是 O(n^2)。KV Cache 缓存已计算的 K/V，每步只需计算新 token 的 Q/K/V 并追加。

**面试连接**：后训练中，reward model 需要对完整 response 打分，如果没有 KV Cache 优化，推理会非常慢，影响 RLHF 训练效率。

#### 3.1.2 PagedAttention 的核心思想

传统做法：为每个请求预分配连续内存 → 内部碎片 + 外部碎片 → 内存浪费。

PagedAttention（vLLM 提出）：将 KV Cache 切分成固定大小的 **Page**（类似操作系统虚拟内存），通过 **Page Table** 做逻辑到物理的映射。

**Mini-SGLang 实现**（`kvcache/mha_pool.py:10-68`）：
```python
# KV Cache 的物理存储形状
# (2, num_layers, num_pages, page_size, num_kv_heads, head_dim)
#  2 = key + value
self._kv_buffer = torch.empty(
    (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
    device=device, dtype=dtype,
)
```

**Page Table**（`engine/engine.py:69-73`）：
```python
# 每个请求一行，每列对应一个 token 位置，值是物理存储位置
# page_size=1 时，直接存 token 位置；page_size>1 时，存 page 起始位置
self.page_table = torch.zeros(
    (config.max_running_req + 1, aligned_max_seq_len),
    dtype=torch.int32, device=self.device,
)
```

**Nano-vLLM 实现**（`engine/block_manager.py`）：
- 使用 Block（固定大小 256 token）管理 KV Cache
- 每个 Block 有 `ref_count`、`hash`、`token_ids` 用于 prefix caching
- `hash_to_block_id` 字典实现 hash → block 的快速查找

**面试可深挖的点**：
1. **Page Size 的影响**：Mini-SGLang 支持可配置 page_size（1/16/32/64），Nano-vLLM 固定 256。page_size 小 → 粒度细、浪费少，但 page table 大；page_size 大 → 对齐效率高但内部碎片多。
2. **Page Table 的设计**：Mini-SGLang 的 page table 始终按 page_size=1 对齐（`core.py:103`），需要时再做除法转换。这是一种工程 trick，简化了索引逻辑。
3. **与 CUDA Graph 的冲突**：CUDA Graph 要求固定形状，但 PagedAttention 的 page table 是动态的 → 需要 capture 时预分配 buffer，replay 时 copy 新数据。

#### 3.1.3 面试问题：KV Cache 的内存怎么算？

```
KV Cache 大小 = 2(K+V) × num_layers × num_kv_heads × head_dim × seq_len × dtype_bytes
```

例如 Qwen3-14B（`num_layers=40, num_kv_heads=8, head_dim=128, bf16=2bytes`）：
- 单条 4096 token 的 KV Cache = 2 × 40 × 8 × 128 × 4096 × 2 = **640 MB**

这就是为什么推理框架要精确管理 KV Cache 内存。

---

### 3.2 Prefix Caching（前缀缓存）

#### 3.2.1 为什么重要

在实际应用中，很多请求共享相同的前缀：
- **System Prompt**：同一个 Agent 的所有请求共享系统提示
- **Few-shot Examples**：RAG 场景中，检索到的文档片段被多个请求复用
- **Multi-turn Dialogue**：对话历史是共享前缀

Prefix Caching 可以跳过已缓存前缀的计算，**节省 50-90% 的 prefill 计算**。

#### 3.2.2 两种实现方式对比

**Mini-SGLang 的 Radix Cache**（`kvcache/radix_cache.py`）：

使用 **Radix Tree**（基数树）管理前缀：
- 每个节点存储一段 token 序列（key）和对应的物理位置（value）
- 共享前缀的请求可以复用同一节点的 KV Cache
- 支持 LRU 淘汰（按 timestamp 排序的最小堆）
- 引用计数（ref_count）区分 evictable 和 protected 的缓存

```python
class RadixTreeNode:
    children: Dict[Any, RadixTreeNode]  # 子节点
    ref_count: int                       # 引用计数
    timestamp: int                       # LRU 时间戳
    _key: torch.Tensor                   # token ids
    _value: torch.Tensor                 # 物理存储位置
```

**匹配流程**（`_tree_walk` 方法）：
1. 从根节点开始，逐层查找匹配的子节点
2. 找到子节点后，比较 key 的前缀匹配长度
3. 如果部分匹配，执行 `split_at` 拆分节点
4. 返回匹配的节点和前缀长度

**Nano-vLLM 的 Block-level Prefix Caching**（`engine/block_manager.py`）：

使用 **Hash 链** 管理：
- 每个 Block 计算一个 hash（基于 token_ids + 前一个 block 的 hash）
- `hash_to_block_id` 字典做全局查找
- 更简单但粒度更粗（block 级别，256 token）

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

**面试可深挖的点**：

1. **Radix Tree vs Hash Chain**：
   - Radix Tree 支持任意长度前缀匹配，Hash Chain 只能按 block 对齐
   - Radix Tree 的 split 操作可以处理部分匹配，Hash Chain 不行
   - Hash Chain 实现更简单，查找 O(1)，Radix Tree 是 O(树深度)

2. **Eviction 策略**：
   - Mini-SGLang 用 LRU（最近最少使用），按 timestamp 排序
   - 实际生产中还可以考虑 LFU、优先级等策略
   - 引用计数 > 0 的节点不能被 evict（正在被使用）

3. **与 Agent 场景的关系**：
   - Agent 的 tool-use 调用中，system prompt + tool definitions 是共享前缀
   - 多轮对话中，历史消息是共享前缀
   - 好的 prefix caching 可以让 Agent 的响应延迟大幅降低

---

### 3.3 Chunked Prefill（分块预填充）

#### 3.3.1 问题背景

长 prompt 的 prefill 阶段需要一次性处理所有 input tokens，可能导致：
- **峰值显存爆炸**：长序列的中间激活值占用大量显存
- **Decode 延迟抖动**：长 prefill 阻塞了正在 decode 的请求

#### 3.3.2 解决方案

将长 prompt 分成多个 chunk，每个 chunk 独立做 prefill，中间插入 decode 步骤。

**Mini-SGLang 实现**（`scheduler/prefill.py:23-90`）：

```python
class ChunkedReq(Req):
    def append_host(self, next_token):
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # 避免被加入 decode batch
```

关键设计：
- `PrefillAdder` 有 `token_budget` 限制，每个 chunk 最多处理 `max_extend_tokens` 个 token
- 如果一次 prefill 不完，创建 `ChunkedReq`，下次继续
- `ChunkedReq` 的 `can_decode` 返回 False，不会被误加入 decode batch

**Nano-vLLM 也支持**（`scheduler.py:42-43`）：
```python
if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
    break
```

**面试连接**：
- Chunked Prefill 对 **长上下文 Agent** 场景至关重要（如 Claude 的 200K context）
- 后训练中，如果 prompt 很长（如 CoT 数据），chunked prefill 可以避免 OOM
- 可以和 **Continuous Batching** 结合，在 prefill 间隙服务 decode 请求

---

### 3.4 Continuous Batching（连续批处理）

#### 3.4.1 Static Batching 的问题

传统做法：一批请求必须全部完成才能开始下一批。短请求要等最长的请求完成 → GPU 空闲。

#### 3.4.2 Continuous Batching

每个 step 可以动态增删请求：
- 有新请求进来 → 下一个 step 就可以加入
- 有请求生成完毕 → 下一个 step 就可以移除

**Mini-SGLang 的实现**（`scheduler/scheduler.py:219-225`）：
```python
def _schedule_next_batch(self) -> ForwardInput | None:
    # Prefill 优先，然后 Decode
    batch = (
        self.prefill_manager.schedule_next_batch(self.prefill_budget)
        or self.decode_manager.schedule_next_batch()
    )
    return self._prepare_batch(batch) if batch else None
```

**Nano-vLLM 的实现**（`engine/scheduler.py:25-73`）：
- 先尝试 schedule prefill（从 waiting 队列取）
- 如果没有 prefill，再 schedule decode（从 running 队列取）
- 支持 preemption（抢占）：显存不够时，把最晚的 decode 请求踢回 waiting

**面试可深挖的点**：

1. **Prefill vs Decode 的调度策略**：
   - Mini-SGLang：Prefill 优先（减少排队延迟）
   - 可以设计为 Decode 优先（减少已开始请求的延迟）
   - 还可以做 Fair Scheduling（按比例分配）

2. **Preemption 策略**：
   - Nano-vLLM 用简单的 FIFO（踢掉最后加入的）
   - 可以考虑优先级、已完成比例等更复杂的策略
   - Preemption 意味着之前计算的 KV Cache 可能浪费

---

### 3.5 Overlap Scheduling（重叠调度）

#### 3.5.1 核心思想

在 GPU 执行当前 batch 的 forward 时，CPU 同时准备下一个 batch 的 metadata（page table、position ids 等）。这样 CPU 开销被 GPU 计算"隐藏"。

**Mini-SGLang 实现**（`scheduler/scheduler.py:83-106`）：
```python
def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
    # 1. 接收新消息
    for msg in self.receive_msg(blocking=blocking):
        self._process_one_msg(msg)

    # 2. 在 CPU 上准备下一个 batch
    forward_input = self._schedule_next_batch()

    # 3. 在 GPU stream 上启动 forward（异步）
    ongoing_data = None
    if forward_input is not None:
        with self.engine_stream_ctx:
            self.engine.stream.wait_stream(self.stream)
            ongoing_data = (forward_input, self._forward(forward_input))

    # 4. 处理上一个 batch 的结果（CPU 上）
    self._process_last_data(last_data)

    return ongoing_data
```

使用两个 CUDA Stream：
- `self.stream`：CPU 准备 metadata
- `self.engine.stream`：GPU 执行 forward
- 通过 `wait_stream` 同步

**面试连接**：这和训练框架中的 **DataLoader prefetch** 是同一个思想。后训练中，如果 reward model 的推理是瓶颈，overlap scheduling 可以提升吞吐。

---

### 3.6 CUDA Graph

#### 3.6.1 问题

Decode 阶段，batch size 通常较小（< 256），每次 forward 只做少量计算，但 **CPU kernel launch 开销** 成为瓶颈。CUDA Graph 将一系列 kernel 操作录制下来，之后直接 replay，避免 CPU launch 开销。

#### 3.6.2 Mini-SGLang 实现（`engine/graph.py:78-171`）

```python
class GraphRunner:
    def _capture_graphs(self, ...):
        for bs in pbar:
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            # 预热
            self.buffer.logits[:bs] = model.forward()
            # 捕获
            with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                self.buffer.logits[:bs] = model.forward()
            self.graph_map[bs] = graph

    def replay(self, batch: Batch) -> torch.Tensor:
        self.buffer.copy_from(batch)  # 复制新数据到 buffer
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()  # 直接 replay
        return self.buffer.logits[:batch.size]
```

**关键设计**：
- 预捕获多个 batch size 的 graph（[1, 2, 4, 8, 16, 24, ...]）
- 运行时选择最近的 bs 做 padding（`pad_batch`）
- 使用 `graph.pool()` 复用 CUDA 内存池，减少显存占用
- 只用于 decode 阶段（prefill 的序列长度不固定，无法用 graph）

**Nano-vLLM 实现**（`engine/model_runner.py:222-257`）：
- 类似思路，但用 dict 存储 graph_vars
- 用 `self.graph_vars` 做 buffer 管理

**面试可深挖的点**：

1. **CUDA Graph 的限制**：
   - 只能 capture 固定形状的计算
   - 不支持 dynamic control flow（如 if/else）
   - 需要预分配所有 buffer

2. **与 torch.compile 的关系**：
   - torch.compile 是编译时优化，CUDA Graph 是运行时优化
   - Nano-vLLM 的 Sampler 用了 `@torch.compile`（`layers/sampler.py:7`）
   - 两者可以结合使用

---

### 3.7 Tensor Parallelism（张量并行）

#### 3.7.1 基本思想

将模型的权重矩阵按列或行切分到多个 GPU 上：
- **Column Parallel**：权重按列切分，每个 GPU 计算部分输出
- **Row Parallel**：权重按行切分，每个 GPU 计算部分输入的加权和，最后 all-reduce

#### 3.7.2 Mini-SGLang 实现（`layers/linear.py`）

```python
class LinearQKVMerged(_LinearTPImpl):
    """QKV 合并投影，按列切分"""
    def __init__(self, ...):
        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        # 每个 GPU 只存储 local 部分的权重

class LinearOProj(_LinearTPImpl):
    """Output 投影，按行切分"""
    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)  # 需要 all-reduce
        return y
```

**关键设计**：
- QKV Projection：Column Parallel（每个 GPU 计算部分 head）
- Output Projection：Row Parallel + All-Reduce
- MLP 的 Gate/Up：Column Parallel（合并为一个大矩阵）
- MLP 的 Down：Row Parallel + All-Reduce
- MoE 的 Expert 也可以按 TP 切分

**分布式通信**（`distributed/impl.py`）：
```python
class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x):
        return self.plugins[-1].all_reduce(x)
```

支持两种后端：
- `TorchDistributedImpl`：用 PyTorch 原生 `dist.all_reduce`
- `PyNCCLDistributedImpl`：用 PyNCCL（更高效，支持异步）

**面试连接**：
- TP 主要用于大模型（70B+），单卡放不下
- 后训练中，如果要做大规模 RLHF，TP + DP 的混合并行是必须的
- GQA（Grouped Query Attention）中，KV heads 比 Q heads 少，TP 时需要特殊处理（`allow_replicate=True`）

---

### 3.8 Attention Backend 优化

#### 3.8.1 FlashAttention vs FlashInfer

Mini-SGLang 支持三种 attention 后端：

| 后端 | 特点 | 适用场景 |
|------|------|----------|
| FlashAttention (FA) | 通用、稳定 | Prefill 阶段 |
| FlashInfer (FI) | Paged KV Cache 原生支持 | Decode 阶段 |
| TRT-LLM FMHA | NVIDIA 官方优化 | Hopper GPU |

**HybridBackend**（`attention/base.py:37-63`）：
```python
class HybridBackend(BaseAttnBackend):
    def __init__(self, prefill_backend, decode_backend):
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(self, q, k, v, layer_id, batch):
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)
```

可以为 prefill 和 decode 使用不同的后端（如 Hopper GPU 上默认 FA3 做 prefill，FI 做 decode）。

#### 3.8.2 KV Cache 存储与读取

```python
# 存储新计算的 K/V 到 cache
self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

# 从 cache 读取所有 K/V 做 attention
return _fa_sgl_impl(
    q=q,
    k_cache=self.kvcache.k_cache(layer_id),
    v_cache=self.v_cache(layer_id),
    page_table=metadata.page_table,
    ...
)
```

**Nano-vLLM 用 Triton kernel 做存储**（`layers/attention.py:10-40`）：
```python
@triton.jit
def store_kvcache_kernel(key_ptr, ..., slot_mapping_ptr, D: tl.constexpr):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    # 读取 key/value，写入 cache 对应 slot
```

**面试可深挖的点**：

1. **Prefill vs Decode 的 Attention 差异**：
   - Prefill：一个请求处理多个 token → varlen attention（变长）
   - Decode：多个请求各处理 1 个 token → batched attention
   - 两者的 memory access pattern 完全不同

2. **Paged KV Cache 的 Attention 实现**：
   - 需要 page_table 做间接寻址
   - FlashAttention 的 `block_table` 参数就是为此设计
   - FlashInfer 的 `paged_kv_indices` 做同样的事

---

### 3.9 Sampling（采样）

#### 3.9.1 采样策略

```python
# Mini-SGLang 的 Sampler（engine/sample.py:24-45）
def sample_impl(logits, temperatures, top_k, top_p):
    probs = sampling.softmax(logits, temperatures)
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)
    if top_p is None:
        return sampling.top_k_sampling_from_probs(probs, top_k)
    if top_k is None:
        return sampling.top_p_sampling_from_probs(probs, top_p)
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)
```

**Nano-vLLM 的 Sampler** 用 `@torch.compile` 优化：
```python
@torch.compile
def forward(self, logits, temperatures):
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    # Gumbel-max trick: 等价于 multinomial sampling
    sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
    return sample_tokens
```

**面试连接**：
- **Temperature**：控制随机性。temperature → 0 等价于 greedy，temperature → ∞ 等价于均匀分布
- **Top-p (Nucleus Sampling)**：只从累积概率达到 p 的最小 token 集合中采样
- **Top-k**：只从概率最高的 k 个 token 中采样
- **Greedy**：直接 argmax，确定性输出

后训练中，采样策略直接影响：
- **SFT 数据生成**：通常用 temperature > 0 来增加多样性
- **RLHF/GRPO**：需要 temperature > 0 来探索
- **Evaluation**：通常用 greedy 或 temperature = 0

---

## 四、代码模块详解（面试中如何介绍项目）

### 4.1 模块依赖关系

```
minisgl/
├── core.py           # Req, Batch, Context, SamplingParams（核心数据结构）
├── server/           # API Server + CLI
│   ├── api_server.py # FastAPI, OpenAI 兼容 API
│   ├── args.py       # 命令行参数
│   └── launch.py     # 多进程启动
├── tokenizer/        # Tokenizer/Detokenizer Worker
├── message/          # 进程间消息定义（ZMQ 序列化）
├── scheduler/        # 调度器（核心）
│   ├── scheduler.py  # 主循环 + Overlap Scheduling
│   ├── prefill.py    # Prefill 管理 + Chunked Prefill
│   ├── decode.py     # Decode 管理
│   ├── cache.py      # CacheManager（KV Cache 分配/回收）
│   ├── table.py      # TableManager（Page Table 管理）
│   └── io.py         # Scheduler I/O（ZMQ 通信）
├── engine/           # 推理引擎
│   ├── engine.py     # Engine（模型 + KVCache + Backend）
│   ├── graph.py      # CUDA Graph Runner
│   └── sample.py     # Sampler（采样逻辑）
├── attention/        # Attention 后端
│   ├── base.py       # BaseAttnBackend, HybridBackend
│   ├── fa.py         # FlashAttention 后端
│   └── fi.py         # FlashInfer 后端
├── kvcache/          # KV Cache 管理
│   ├── base.py       # BaseKVCachePool, BasePrefixCache
│   ├── mha_pool.py   # MHAKVCache（物理存储）
│   ├── radix_cache.py # RadixPrefixCache（前缀缓存）
│   └── naive_cache.py # NaivePrefixCache（无缓存）
├── layers/           # 模型层
│   ├── linear.py     # TP Linear 层
│   ├── attention.py  # AttentionLayer（调用 attn backend）
│   ├── rotary.py     # RoPE
│   ├── norm.py       # RMSNorm
│   ├── embedding.py  # Embedding
│   └── moe.py        # MoE Layer
├── models/           # 模型实现
│   ├── qwen3.py      # Qwen3
│   ├── qwen3_moe.py  # Qwen3-MoE
│   ├── llama.py      # Llama
│   └── weight.py     # 权重加载 + 切分
├── moe/              # MoE 后端
│   └── fused.py      # Fused MoE Kernel
├── kernel/           # 自定义 CUDA/Triton kernel
│   ├── radix.py      # Radix Tree 比较 kernel
│   ├── store.py      # KV Cache 存储 kernel
│   └── pynccl.py     # PyNCCL 封装
└── distributed/      # 分布式通信
    ├── impl.py       # 通信实现
    └── info.py       # TP Rank/Size 信息
```

### 4.2 请求生命周期（面试讲故事用）

```
1. 用户发 POST /v1/chat/completions
2. API Server 创建 TokenizeMsg，发给 Tokenizer Worker
3. Tokenizer 做 tokenize，创建 UserMsg，发给 Scheduler (Rank 0)
4. Scheduler 收到消息，加入 PrefillManager 的 pending_list
5. _schedule_next_batch() 选择 prefill batch：
   a. CacheManager.match_req() → 在 Radix Tree 中查找前缀
   b. PrefillAdder._try_allocate_one() → 分配 Page Table 槽位
   c. 如果 input 太长，创建 ChunkedReq 分块处理
6. Scheduler._prepare_batch() 准备 forward 数据：
   a. CacheManager.allocate_paged() → 分配物理 pages
   b. 构造 positions, input_mapping, write_mapping
   c. AttentionBackend.prepare_metadata() → 构造 attention 元数据
7. Engine.forward_batch() 执行 forward：
   a. 如果可以用 CUDA Graph → GraphRunner.replay()
   b. 否则 → model.forward()
   c. Sampler.sample() → 采样下一个 token
8. Scheduler._process_last_data() 处理结果：
   a. 检查是否生成 EOS 或达到 max_tokens
   b. 如果完成，释放资源
   c. 如果是 prefill 阶段，缓存前缀到 Radix Tree
   d. 发送 DetokenizeMsg 给 Detokenizer
9. Detokenizer 做 detokenize，发给 API Server
10. API Server 返回给用户（支持 streaming）
```

---

## 五、面试深度问题清单

### 5.1 基础概念题

**Q1: 什么是 Continuous Batching？它和 Static Batching 有什么区别？**

答：Static Batching 中，一批请求必须全部完成才能开始下一批，短请求要等最长的完成。Continuous Batching 允许每个 step 动态增删请求。Mini-SGLang 中，`_schedule_next_batch()` 每次都重新选择 batch，新请求可以在下一个 step 加入。

**Q2: 什么是 Prefix Caching？在哪些场景下最有价值？**

答：Prefix Caching 缓存已计算的 KV Cache 前缀，后续请求如果共享相同前缀可以直接复用。最有价值的场景：
- Agent 的 system prompt 共享
- Multi-turn dialogue 的历史共享
- RAG 中相同文档片段的共享
- Few-shot prompt 共享

Mini-SGLang 用 Radix Tree 实现，支持任意长度前缀匹配。

**Q3: 什么是 PagedAttention？为什么比连续内存分配好？**

答：PagedAttention 将 KV Cache 切分为固定大小的 Page，通过 Page Table 做逻辑到物理映射。好处：
1. 消除内部碎片（最后一个 page 可能不满）
2. 消除外部碎片（page 可以任意分配）
3. 支持 prefix caching（共享 page）
4. 支持 memory overcommitment（按需分配）

### 5.2 深挖实现题

**Q4: Radix Cache 的 eviction 策略是什么？如何保证正在使用的缓存不被驱逐？**

答：Mini-SGLang 的 Radix Cache 用 **LRU + 引用计数**：
- 每个节点有 `ref_count`，正在使用的节点 ref_count > 0
- 只有 ref_count == 0 且是叶子节点的才能被驱逐
- 用最小堆按 timestamp 排序，驱逐最久未使用的
- 驱逐时从叶子向根传播：如果父节点驱逐后变成叶子且 ref_count == 0，也可以被驱逐

**Q5: Chunked Prefill 如何实现？Chunk 边界处的 Attention 怎么处理？**

答：Mini-SGLang 的实现：
1. `PrefillAdder` 有 `token_budget`，每个 chunk 最多处理 budget 个 token
2. 如果一次 prefill 不完，创建 `ChunkedReq`（`can_decode = False`）
3. 下次调度时，`ChunkedReq` 继续从上次中断处 prefill
4. Attention 层面，每个 chunk 独立做 extend attention，cu_seqlens_q 和 cu_seqlens_k 处理边界

**Q6: CUDA Graph 如何与 PagedAttention 配合？Page Table 是动态的怎么办？**

答：
1. Capture 时用 dummy 数据预分配固定大小的 buffer
2. Replay 前，将真实数据 copy 到 buffer 中（`buffer.copy_from(batch)`）
3. Page Table 的更新在 `prepare_for_replay()` 中完成
4. 使用 `graph.pool()` 复用内存池，减少显存占用

**Q7: Tensor Parallelism 中，GQA 的 KV heads 比 Q heads 少，怎么处理？**

答：Mini-SGLang 用 `div_even(num_kv_heads, tp_size, allow_replicate=True)`。当 KV heads 不能被 TP size 整除时，允许 replicate（复制）。例如 8 KV heads，TP=3，则每个 GPU 分到 3 heads（最后一个 GPU 多一个）。

### 5.3 架构设计题

**Q8: 如果你要支持 Speculative Decoding（投机解码），需要改哪些地方？**

答：需要改：
1. **Scheduler**：支持 draft model 的 forward，以及 verify step
2. **Engine**：增加 draft model 的 forward_batch
3. **Batch**：支持同时包含 draft tokens 和 verify tokens
4. **Sampler**：实现 reject sampling 或 accept/reject 策略
5. **KV Cache**：支持 draft tokens 的临时缓存和回滚

**Q9: 如何设计一个支持多 LoRA 的推理系统？**

答：
1. **LoRA 权重管理**：每个 LoRA 有独立的 A/B 矩阵，需要按需加载
2. **Batch 调度**：同一 batch 中的请求可能用不同 LoRA → 需要 per-request LoRA index
3. **Linear 层改造**：`y = Wx + BAx`，其中 B/A 是 LoRA 矩阵
4. **KV Cache**：LoRA 不影响 KV Cache（只是改了投影矩阵）
5. **CUDA Graph**：不同 LoRA 的计算图相同，只需要更新权重 buffer

**Q10: 如何让推理系统支持 Agent 的 Function Calling？**

答：
1. **Tokenizer**：需要特殊 token 处理（如 `<tool_call>` 标签）
2. **Sampling**：可能需要 constrained decoding（只允许生成合法的 JSON）
3. **Streaming**：需要在流式输出中检测 tool call 的边界
4. **Prefix Caching**：tool definitions 是共享前缀
5. **Stop Condition**：除了 EOS，还需要支持自定义 stop tokens

### 5.4 后训练相关题

**Q11: RLHF 训练中，reward model 的推理和 policy model 的推理有什么不同？**

答：
1. **输入不同**：reward model 输入完整 response，policy model 是自回归生成
2. **Batching 不同**：reward model 可以 static batch（输入长度相似），policy model 需要 continuous batch
3. **KV Cache**：reward model 通常不需要 KV Cache（只需要最后一层 hidden state），policy model 需要
4. **生成 vs 评分**：reward model 是 forward-only，policy model 需要 generate

**Q12: 如何用推理框架的 prefix caching 优化 online RLHF？**

答：Online RLHF 中，多个采样请求共享相同的 prompt → 可以用 prefix caching 复用 prompt 的 KV Cache。具体：
1. 先对 prompt 做一次 prefill，缓存到 Radix Tree
2. 后续采样请求直接复用 prompt 的 KV Cache
3. 不同 response 部分各自独立 compute
4. 可以节省 50%+ 的 prefill 计算

---

## 六、面试中的项目介绍模板

### 6.1 一句话介绍

> "我研读了 SGLang 和 vLLM 两个主流推理框架的精简实现（Mini-SGLang ~5000行，Nano-vLLM ~1200行），深入理解了 LLM serving 的核心优化：PagedAttention、Prefix Caching、Chunked Prefill、Continuous Batching、CUDA Graph、Tensor Parallelism 等。"

### 6.2 详细介绍（2-3 分钟）

> "我系统学习了 LLM 推理框架的实现。以 Mini-SGLang 为例，它的核心是 **Scheduler + Engine** 两层设计：
>
> **Scheduler 层**负责请求调度：收到新请求后，先在 Radix Cache 中查找共享前缀，然后决定是做 prefill 还是 decode。对于长 prompt，支持 chunked prefill 分块处理。调度策略是 prefill 优先，同时支持 overlap scheduling 来隐藏 CPU 开销。
>
> **Engine 层**负责模型执行：管理模型权重、KV Cache 池、Attention 后端。支持 CUDA Graph 加速 decode，支持 FlashAttention 和 FlashInfer 两种 attention 后端，支持 TP 多卡并行。
>
> 我还对比了 Nano-vLLM（vLLM 的精简版），发现两者的核心思想一致，但在 prefix caching 上有不同设计：Mini-SGLang 用 Radix Tree 支持任意长度匹配，Nano-vLLM 用 block-level hash chain。
>
> 这些知识帮助我理解了后训练中的 infra 瓶颈：比如 RLHF 中 reward model 的推理效率、Agent 场景中 prefix caching 对延迟的影响、长上下文场景中 chunked prefill 的必要性等。"

### 6.3 面试官追问应对

如果面试官问 **"你实际改过这些代码吗？"**

诚实回答："我主要是研读和理解。但我尝试了以下实验：
1. 修改 page_size 观察对吞吐的影响
2. 关闭 overlap scheduling 做 ablation study
3. 对比 radix cache vs naive cache 的性能差异
4. 阅读了相关的论文（Sarathi-Serve, NanoFlow, FlashAttention, SGLang）"

---

## 七、关键论文与延伸阅读

| 论文 | 关键贡献 | 与项目的关系 |
|------|----------|-------------|
| [FlashAttention 1/2/3](https://arxiv.org/abs/2205.14135) | IO-aware Attention | Mini-SGLang 的 FA 后端 |
| [vLLM (PagedAttention)](https://arxiv.org/abs/2309.06180) | Paged KV Cache | Nano-vLLM 的理论基础 |
| [SGLang](https://arxiv.org/abs/2312.07104) | Radix Attention | Mini-SGLang 的直接上游 |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | Chunked Prefill | Mini-SGLang 的 prefill 策略 |
| [NanoFlow](https://arxiv.org/abs/2408.12757) | Overlap Scheduling | Mini-SGLang 的调度策略 |
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | MLA + MoE | MoE 支持的动机 |
| [FlashInfer](https://arxiv.org/abs/2501.01005) | Paged KV Cache Attention | Mini-SGLang 的 FI 后端 |

---

## 八、与后训练/Agent 的连接点（面试加分项）

### 8.1 推理效率与 RLHF/GRPO

- **Online RLHF** 需要频繁从 policy model 采样 → 推理吞吐直接影响训练速度
- **Prefix Caching** 可以复用 prompt 的 KV Cache → 多个采样请求共享 prompt 计算
- **Continuous Batching** 可以动态调度采样请求 → 提高 GPU 利用率
- **Chunked Prefill** 可以处理长 prompt（如 CoT 数据） → 避免 OOM

### 8.2 推理约束与 Constrained Decoding

- Agent 的 Function Calling 需要 **结构化输出**（JSON）
- 推理框架需要支持 **token masking**（只允许生成合法 token）
- 这影响 Sampling 层：需要在 softmax 前将非法 token 的 logits 设为 -inf

### 8.3 推理延迟与 Agent 体验

- Agent 的 response time = LLM inference time + tool execution time
- **Speculative Decoding** 可以减少 30-50% 的延迟
- **Prefix Caching** 可以减少 50-90% 的 prompt prefill 时间
- **Streaming** 让用户更早看到部分结果

### 8.4 长上下文与 RAG

- 128K context 的 KV Cache 需要 ~20GB 显存（14B 模型）
- **Chunked Prefill** 避免长 prompt 的 OOM
- **Page-level Prefix Caching** 可以复用检索到的文档片段的 KV Cache

---

## 九、Mini-SGLang vs Nano-vLLM 对比表

| 特性 | Mini-SGLang | Nano-vLLM |
|------|-------------|-----------|
| 代码量 | ~5000 行 | ~1200 行 |
| Prefix Caching | Radix Tree（任意长度） | Block-level Hash（256 token） |
| Chunked Prefill | 完整支持 | 基本支持 |
| Overlap Scheduling | 双 CUDA Stream | 不支持 |
| CUDA Graph | 支持多 bs + padding | 支持多 bs |
| TP 通信 | ZMQ + NCCL/PyNCCL | SharedMemory + NCCL |
| Online API | OpenAI 兼容 | 无 |
| MoE 支持 | Qwen3-MoE + Fused MoE | 不支持 |
| Attention 后端 | FA3/FA4 + FlashInfer + TRT-LLM | FlashAttention |
| 模型支持 | Llama/Qwen2.5/Qwen3/Mistral | Qwen3 |
| Sampling | FlashInfer kernel | torch.compile + Gumbel-max |
| Preemption | 不支持（优雅降级） | 支持（FIFO） |

---

## 十、速查：关键代码位置

| 功能 | Mini-SGLang 文件 | 行号 |
|------|-----------------|------|
| 核心数据结构 (Req/Batch/Context) | `core.py` | 15-136 |
| Scheduler 主循环 | `scheduler/scheduler.py` | 83-131 |
| Prefill 调度 + Chunked Prefill | `scheduler/prefill.py` | 32-162 |
| Decode 调度 | `scheduler/decode.py` | 10-39 |
| Cache Manager (页分配/回收) | `scheduler/cache.py` | 15-146 |
| Radix Cache 实现 | `kvcache/radix_cache.py` | 101-237 |
| KV Cache 物理存储 | `kvcache/mha_pool.py` | 10-68 |
| Engine 初始化 | `engine/engine.py` | 29-210 |
| CUDA Graph 捕获/回放 | `engine/graph.py` | 78-171 |
| Sampler 实现 | `engine/sample.py` | 24-75 |
| FlashAttention 后端 | `attention/fa.py` | 36-182 |
| FlashInfer 后端 | `attention/fi.py` | 80-271 |
| TP Linear 层 | `layers/linear.py` | 13-127 |
| Attention Layer (含 RoPE) | `layers/attention.py` | 18-57 |
| MoE Layer | `layers/moe.py` | 9-59 |
| Fused MoE Kernel | `moe/fused.py` | 127-256 |
| Qwen3 模型定义 | `models/qwen3.py` | 18-83 |
| Qwen3-MoE 模型定义 | `models/qwen3_moe.py` | 18-83 |
| 权重加载 + TP 切分 | `models/weight.py` | - |
| API Server | `server/api_server.py` | 225-452 |
| 多进程启动 | `server/launch.py` | 40-117 |
| Offline LLM 接口 | `llm/llm.py` | 28-98 |

---

## 附录：常见 LLM Serving 面试题速答

1. **Prefill vs Decode**：Prefill 是一次性处理所有 input tokens（计算密集），Decode 是逐 token 生成（访存密集）。
2. **为什么 Decode 是访存密集**：每步只生成 1 个 token，但要读取整个 KV Cache → 算术强度低。
3. **什么是 Continuous Batching**：每 step 动态增删请求，避免短请求等长请求。
4. **什么是 Prefix Caching**：缓存共享前缀的 KV Cache，避免重复计算。
5. **什么是 Speculative Decoding**：用小模型快速生成多个候选 token，大模型一次性验证，减少 decode 步数。
6. **KV Cache 大小怎么算**：2 × num_layers × num_kv_heads × head_dim × seq_len × dtype_bytes。
7. **Page Size 的影响**：小 page → 细粒度、少浪费、大 page table；大 page → 高对齐效率、多内部碎片。
8. **TP vs PP vs DP**：TP 切权重（同层），PP 切层（流水线），DP 切数据（复制模型）。
9. **FlashAttention 为什么快**：减少 HBM 访问，利用 SRAM 做 online softmax，IO-aware。
10. **CUDA Graph 适合什么场景**：小 batch、固定形状、CPU launch 开销大的场景（如 decode）。
