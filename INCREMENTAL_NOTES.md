# 增量学习笔记 —— 从复现 Plan 中新发现的点

> 本文档记录在制定 8×3090 复现计划过程中，对比前两轮文档（INTERVIEW_PREP.md, INTERVIEW_5_CATEGORIES.md）**新发现和深入理解**的点。
> 前两轮侧重"是什么"和"为什么"，本轮侧重"实际能不能跑"和"跑不通怎么办"。

---

## 一、架构兼容性：SM86 的真实约束

### 1.1 前两轮没讲清楚的：后端选择的代码路径

前两轮讲了"FlashAttention 和 FlashInfer 的区别"，但没讲**在不同 GPU 上到底选哪个**。

实际代码路径（`engine/engine.py:218-225`）：

```python
def _adjust_config(config: EngineConfig):
    if config.attention_backend == "auto":
        backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
        override("attention_backend", backend)
```

这个逻辑揭示了一个重要事实：**后端选择是硬编码的架构判断**，不是"哪个更快选哪个"。

| GPU 架构 | SM 版本 | 自动选择 | 可用后端 |
|----------|---------|---------|---------|
| A100 | SM80 | fi | FlashInfer only |
| 3090 | SM86 | fi | FlashInfer only |
| H100/H200 | SM90 | fa,fi | FA3 prefill + FI decode |
| B100/B200 | SM100 | trtllm | TRT-LLM only |

**新洞察**：A100 和 3090 是同一档（都只能用 FlashInfer），但 A100 有 NVLink 而 3090 没有。这意味着 **3090 的 TP 效率比 A100 更差**，不只是算力差距，还有通信差距。

### 1.2 FlashAttention 后端为什么不能用于 SM86

`attention/fa.py` 中的 FlashAttention 后端调用的是 `sgl_kernel.flash_attn.flash_attn_with_kvcache`，这不是标准的 FlashAttention 库，而是 SGLang 团队自己编写的 CUDA kernel，它使用了 SM90 特有的指令（如 TMA、WGMMA）。

```python
# attention/fa.py:157-163
from sgl_kernel.flash_attn import flash_attn_with_kvcache
return flash_attn_with_kvcache(q=q, k_cache=k_cache, v_cache=v_cache, ..., ver=version)
```

而 FlashInfer 的 fa2 后端是纯 CUDA 实现，支持 SM80+：

```python
# attention/fi.py:96-97
self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
    ..., backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
)
```

**新洞察**：FlashInfer 自己也有 fa2 和 fa3 两个后端。代码注释说 `fa3 is slow, use fa2 instead`，这意味着即使是 H100 上，FlashInfer 也默认用 fa2。这是一个**违反直觉**的工程发现——更新的 kernel 不一定更快。

### 1.3 TRT-LLM 后端的 page_size 约束

```python
# engine/engine.py:227-229
if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
    override("page_size", 64)
```

TRT-LLM 只支持 page_size=16/32/64，不支持 page_size=1。这是因为 TRT-LLM 的 kernel 实现对 page 大小有硬性要求。

**新洞察**：不同后端对 page_size 的约束不同：
- FlashInfer: page_size=1 only（`attention/fi.py:66`: `assert self.page_size == 1`）
- TRT-LLM: page_size=16/32/64 only
- FlashAttention: 任意 page_size

这是一个**设计上的 trade-off**：FlashInfer 只支持 page_size=1（最细粒度，但 page table 大），TRT-LLM 只支持大 page（page table 小，但有内部碎片）。

---

## 二、CUDA 13.1 的兼容性挑战

### 2.1 前两轮没涉及的：实际安装会遇到什么问题

在制定复现计划时，发现系统的 CUDA 版本是 13.1（2025/2026 年的新版本），这带来了一系列兼容性问题：

1. **PyTorch**：官方 wheel 通常只编译到 CUDA 12.x。CUDA 13.1 可能需要 nightly 版本或从源码编译。
2. **sgl_kernel**：预编译 wheel 可能不包含 CUDA 13.1 版本。
3. **FlashInfer**：预编译 wheel 可能不包含 CUDA 13.1 版本。
4. **tvm_ffi**：JIT 编译依赖 nvcc，CUDA 13.1 的 nvcc 编译出的 kernel 应该能工作。

**新洞察**：推理框架的**安装难度**往往被低估。一个"pip install"就能解决的事情，在实际环境中可能需要：
- 从源码编译 CUDA extension
- 修改 `setup.py` 中的 `arch` flag
- 调整 CUDA toolkit 版本
- 使用 Docker 绕过环境问题

这是**工程落地能力**的一个具体体现。

### 2.2 JIT 编译 vs 预编译

Mini-SGLang 的 kernel 有两种编译方式（`kernel/utils.py`）：

```python
def load_aot(...):    # AOT = Ahead Of Time，预编译
    from tvm_ffi.cpp import load
    ...

def load_jit(...):    # JIT = Just In Time，运行时编译
    from tvm_ffi.cpp import load_inline
    ...
```

- `store_cache` 用 JIT 编译（`kernel/store.py`）：运行时根据 element_size 动态编译
- `fast_compare_key` 用 AOT 编译（`kernel/radix.py`）：安装时编译

**新洞察**：JIT 编译的好处是**自适应**——可以根据当前 GPU 架构自动选择最优的编译 flag。但坏处是**首次运行慢**（需要几分钟编译），且依赖 nvcc 在 PATH 中。

---

## 三、显存预算的精确计算

### 3.1 前两轮只给了公式，这里给出实际数字

以 Qwen3-14B 为例，TP=2，RTX 3090 (24GB)：

```
模型权重: 14B × 2 bytes / 2 GPU = 14GB per GPU
KV Cache 计算:
  cache_per_page = 2(K+V) × head_dim × (num_kv_heads/tp_size) × page_size × dtype × num_layers
                 = 2 × 128 × (8/2) × 1 × 2 × 40
                 = 40,960 bytes per page
  如果 memory_ratio=0.9, 模型占 14GB, 剩余 24×0.9 - 14 = 7.6GB
  num_pages = 7.6GB / 40,960 = ~194,000 pages
  num_tokens = 194,000 × 1 = 194,000 tokens (总 KV Cache 容量)
```

**新洞察**：24GB 的 3090 放 Qwen3-14B (TP=2) 后，只剩 ~7.6GB 给 KV Cache，约 194K tokens。如果每个请求平均 2K tokens，最多同时服务 ~97 个请求。这个数字比想象中低。

### 3.2 _sync_get_memory 的跨卡对齐检查

```python
# engine/engine.py:170-189
def _sync_get_memory(self):
    free_memory = get_free_memory(self.device)
    free_mem_tensor = torch.tensor([free_memory, -free_memory], ...)
    torch.distributed.all_reduce(free_mem_tensor, op=ReduceOp.MIN, group=self.tp_cpu_group)
    min_free_memory = int(free_mem_tensor[0].item())
    max_free_memory = -int(free_mem_tensor[1].item())
    if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
        raise RuntimeError("Memory across TP ranks are imbalanced")
```

这个检查确保所有 TP rank 的可用显存差异不超过 2GB。如果超过，说明某张卡被其他进程占用，会导致 KV Cache 分配不均。

**新洞察**：这是一个**生产环境的防护机制**。在共享 GPU 集群中，经常会有多人共用 GPU 的情况，如果某张卡被其他任务占用了一部分显存，这个检查会提前报错，而不是等到运行时 OOM。

---

## 四、FlashInfer 的 plan/initialize 机制

### 4.1 前两轮没深入的：FlashInfer 的两阶段调用

FlashInfer 的调用不是直接 forward，而是分两阶段：

1. **plan**：在 CPU 上准备 metadata（cu_seqlens, indices 等），做 H2D copy
2. **run**：在 GPU 上执行 attention kernel

```python
# attention/fi.py:123-166
def _initialize_metadata_once(self, metadata: FIMetadata):
    if metadata.initialized:
        return
    metadata.initialized = True
    # FlashInfer planning reuses a pinned host staging buffer and launches an
    # async H2D copy. Wait here before the next plan mutates that host buffer.
    self.last_event.synchronize()
    metadata.wrapper.plan(...)  # CPU 准备 + H2D copy
    self.last_event.record()
```

**新洞察**：
- `plan` 操作会复用一个 pinned host buffer 做 H2D copy，所以**不能并发调用 plan**（需要 event 同步）
- `initialized` flag 确保每个 metadata 只 plan 一次
- 这就是为什么 FlashInfer 需要 `CUDAGraphBatchDecodeWithPagedKVCacheWrapper` 专门的 graph wrapper——graph capture 时不能调用 plan

### 4.2 FlashInfer 的 workspace buffer 共享

```python
# attention/fi.py:90-107
self.float_workspace_buffer = torch.empty(128 * 1024 * 1024, ...)  # 128MB
self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(self.float_workspace_buffer, ...)
self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(self.float_workspace_buffer, ...)
# HACK: reuse the int_workspace_buffer
self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer
```

**新洞察**：prefill 和 decode 的 wrapper 共享同一个 workspace buffer。这是因为**同一时刻只有一种 phase 在运行**（要么 prefill 要么 decode），所以可以安全共享。这是一个内存优化的工程 trick。

---

## 五、Lazy Free 机制

### 5.1 前两轮提到但没深入的：为什么需要 lazy free

```python
# scheduler/cache.py:93-104
@contextmanager
def lazy_free_region(self):
    def lazy_free(indices: torch.Tensor) -> None:
        lazy_free_list.append(indices[:: self.page_size])
    lazy_free_list: List[torch.Tensor] = []
    try:
        self._free = lazy_free
        yield
    finally:
        del self._free
        self.free_slots = torch.cat([self.free_slots] + lazy_free_list)
```

**问题**：在 `_process_last_data` 中，处理每个请求的结果时可能需要释放 page。但如果直接释放（修改 free_slots），会影响后续请求的分配逻辑。

**解决方案**：lazy free——先收集所有需要释放的 indices，最后一次性释放。

**新洞察**：这个模式在数据库中也很常见（WAL、deferred delete）。核心思想是**批量操作比逐个操作更高效**（减少 torch.cat 的次数），同时避免中间状态的不一致。

---

## 六、CUDA Graph 的 pool 复用

### 6.1 前两轮提到但没解释清楚的：pool 是什么

```python
# engine/graph.py:140-143
with torch.cuda.graph(graph, pool=pool, stream=self.stream):
    self.buffer.logits[:bs] = model.forward()
if pool is None:
    pool = graph.pool()  # reuse cuda graph handle to reduce memory
```

**新洞察**：`graph.pool()` 返回的是一个 CUDA 内存池 handle。后续的 graph capture 可以复用同一块 GPU 内存，因为：
1. 同一时刻只有一个 graph 在 replay
2. 不同 bs 的 graph 使用相同的 buffer 结构（只是 slice 不同）
3. 复用 pool 可以避免为每个 bs 分配独立的 GPU 内存

如果没有 pool 复用，capture 100 个不同 bs 的 graph 需要 100 倍的 buffer 内存。有了 pool 复用，只需要 1 倍。

---

## 七、Weight Loading 的 Merge + Shard 逻辑

### 7.1 前两轮没涉及的：权重加载的工程复杂度

`models/weight.py` 中的权重加载逻辑比想象中复杂：

1. **Merge**：HuggingFace 的权重是分开的（q_proj, k_proj, v_proj），需要合并成 qkv_proj
2. **Shard**：TP 模式下，需要按 rank 切分权重
3. **Expert Stack**：MoE 模型的每个 expert 权重需要 stack 成一个 3D tensor

```python
# 权重合并
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}

# TP 切分
def _shard_tensor(key, value, r, n, num_kv_heads):
    if any(key.count(sub) for sub in _SPLIT_DIM_0):  # q/k/v/gate/up
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):  # o/down
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        # Vocab parallel: 按 vocab 维度切分
        return value[vocab_start_idx:vocab_end_idx, :].clone()
```

**新洞察**：
- QKV 合并的原因：一个 fused linear 比三个 separate linear 效率更高（减少 kernel launch）
- Gate+Up 合并的原因同理：SwiGLU 需要 gate 和 up 逐元素相乘，合并后一次 matmul 搞定
- MoE 的 expert 权重是 `(num_experts, intermediate_size, hidden_size)` 的 3D tensor，这样 fused_moe_kernel 可以一次处理所有 expert

---

## 八、MoE 的 Block Size 对齐

### 8.1 前两轮提到了 MoE，但没讲 block 对齐的细节

```python
# moe/fused.py:31-89
def moe_align_block_size(topk_ids, block_size, num_experts):
    """
    将 token 按 expert 分组，每组对齐到 block_size。
    不够的用 padding token 补齐。
    """
```

MoE 的计算不是"每个 expert 独立算"，而是**将所有 token 按 expert 排序后，分块做矩阵乘法**。这要求每个 expert 的 token 数必须是 block_size 的倍数。

**新洞察**：
- 如果某个 expert 只分到 3 个 token，block_size=16，需要 padding 13 个 token → 计算浪费
- `get_default_config` 中，当 `M <= E`（token 数 ≤ expert 数）时，用更小的 block_size=16
- 这就是 MoE 的 **load balancing** 问题：如果 token 分配不均，padding 浪费会很大

---

## 九、实际复现中的关键差异点

### 9.1 前两轮的"理想"vs 复现的"现实"

| 前两轮的描述 | 实际复现的现实 |
|-------------|--------------|
| "FlashAttention 3 用于 prefill" | SM86 不支持 FA3，只能用 FlashInfer |
| "TP 可以线性扩展吞吐" | PCIe 3090 的 TP 效率远低于 NVLink H100 |
| "CUDA Graph 加速 decode" | 首次 capture 需要几分钟，且占用额外显存 |
| "Radix Cache 节省 50-90% 计算" | 只有在有共享前缀时有效，无共享时退化 |
| "Overlap Scheduling 隐藏 CPU 开销" | 只在 CPU-bound 时有效，GPU-bound 时无收益 |
| "pip install 即可" | CUDA 版本、Python 版本、预编译 wheel 的兼容性是大问题 |

### 9.2 从"读代码"到"跑代码"的认知升级

1. **代码中的 TODO/FIXME 是真实存在的技术债**
   - `scheduler/prefill.py:43`: `# TODO: consider host cache match case`
   - `scheduler/prefill.py:46`: `# TODO: better estimate policy`
   - 这些 TODO 说明当前实现还有改进空间，面试时可以提

2. **环境变量是隐藏的配置**
   - `MINISGL_DISABLE_OVERLAP_SCHEDULING=1`：关闭 overlap
   - `MINISGL_FLASHINFER_USE_TENSOR_CORES=0`：关闭 tensor core
   - `MINISGL_PYNCCL_MAX_BUFFER_SIZE=2G`：调整 PyNCCL buffer
   - 这些环境变量在文档中可能没有完整列出，需要读 `env.py`

3. **日志输出是理解运行时行为的关键**
   - `Auto-selected attention backend: fi`：确认后端选择
   - `Free memory before/after`：确认显存使用
   - `Allocating XXXX tokens for KV cache`：确认 KV Cache 容量
   - 这些日志在 `engine.py` 和 `scheduler.py` 中

---

## 十、面试中可以新增的谈资

基于本轮复现分析，可以新增以下面试谈资：

### 10.1 "我实际尝试过部署，遇到了 XXX 问题"

> "我在 3090 上尝试部署 mini-sglang 时，发现 FlashAttention 3 后端不支持 SM86 架构，只能用 FlashInfer。这让我理解了不同 attention 后端对 GPU 架构的依赖——FA3 使用了 SM90 特有的 TMA 指令，而 FlashInfer 的 fa2 后端是纯 CUDA 实现，兼容 SM80+。"

### 10.2 "我发现了一个反直觉的工程细节"

> "FlashInfer 的 fa3 后端反而比 fa2 慢，代码注释里明确写了 `backend='fa2'  # flashinfer fa3 is slow'`。这说明新的 kernel 实现不一定更快，需要实际 benchmark 验证。"

### 10.3 "我理解了理论和实际的差距"

> "理论上 TP=4 应该比 TP=2 快 2 倍，但实际上在 3090（PCIe）上，TP=4 的 all-reduce 通信开销很大，per-GPU 吞吐可能只有 TP=2 的 70-80%。这就是为什么实际部署时需要根据硬件互联方式选择 TP 度数。"

### 10.4 "我计算过显存预算"

> "以 Qwen3-14B 为例，TP=2 放在两张 3090 上，模型权重占 14GB/卡，剩余 ~7.6GB 给 KV Cache，约 194K tokens。如果每个请求平均 2K tokens，最多同时服务 ~97 个请求。这个数字比想象中低，因为 3090 的 24GB 显存比 H200 的 141GB 小很多。"

---

## 总结：三轮文档的递进关系

| 轮次 | 文档 | 视角 | 核心产出 |
|------|------|------|----------|
| 第一轮 | INTERVIEW_PREP.md | "是什么" | 技术概念 + 源码位置 + 代码走读 |
| 第二轮 | INTERVIEW_5_CATEGORIES.md | "怎么答" | 五类面试问题 + 应答框架 + 追问应对 |
| 第三轮 | REPRODUCTION_PLAN.md + 本文档 | "能不能跑" | 硬件约束 + 兼容性分析 + 实操计划 |

三轮递进：**理解 → 表达 → 实践**。
