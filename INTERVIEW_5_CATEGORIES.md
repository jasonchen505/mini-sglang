# 技术面试五类问题应对指南 —— 基于 Mini-SGLang / Nano-vLLM 源码

> 面试不只是回答概念，而是展示你**为什么这么做、遇到什么问题、怎么解决的、值不值得做**。
> 本文档按五类面试能力组织，每类包含：考察本质、应答框架、具体问答示例、常见追问陷阱。

---

## 第一类：底层原理深入理解

### 面试官在考什么

不是考你能不能背定义，而是考你：
- 这个设计**解决什么问题**（problem-driven）
- 这个设计**有什么局限**（trade-off thinking）
- 如果条件变化，**怎么改进**（engineering judgment）

### 应答框架：Problem → Design → Trade-off → Extension

每个技术点都按这个四步框架回答：

```
1. Problem：当时面临什么问题？为什么旧方案不行？
2. Design：这个方案的核心思想是什么？关键代码在哪？
3. Trade-off：这个方案牺牲了什么？在什么场景下会失效？
4. Extension：如果条件变了（更大模型/更多卡/不同场景），怎么改？
```

---

### Q1: PagedAttention 解决什么问题？为什么不能用连续内存？

**Problem**：

传统 KV Cache 为每个请求预分配一块连续内存，大小等于 `max_seq_len`。这导致两个问题：
1. **内部碎片**：实际生成的序列远短于 max_seq_len，预分配的内存大量浪费
2. **外部碎片**：不同请求的 KV Cache 大小不同，释放后内存中出现不连续的空洞，无法被新请求使用

实际中，按 max_seq_len=4096 预分配，但平均生成长度只有 512，内存利用率只有 12.5%。

**Design**：

PagedAttention 将 KV Cache 切分为固定大小的 Page（类似操作系统虚拟内存），通过 Page Table 做逻辑到物理映射。

Mini-SGLang 的实现（`kvcache/mha_pool.py:28-32`）：
```python
# 物理存储：(2, num_layers, num_pages, page_size, num_kv_heads, head_dim)
# 逻辑访问：通过 page_table[table_idx, position] → 物理位置
self._kv_buffer = torch.empty(
    (2, num_layers, num_pages, page_size, local_kv_heads, head_dim), ...)
```

Page Table（`engine/engine.py:69-73`）每个请求一行，每列对应一个 token 位置，值是物理存储的 slot index。分配时从 free_slots 池中取，释放时还回去。

**Trade-off**：

1. **Page Table 本身占用显存**：page_size=1 时，page table 和 token 数一样大。Mini-SGLang 的 page_table 形状是 `(max_running_req+1, aligned_max_seq_len)`，dtype=int32，对于 1000 并发 × 4096 长度 = 16MB。不大，但不是零。
2. **页对齐浪费**：最后一个页可能没填满，page_size=16 时平均浪费 8 个 token 的 KV Cache。
3. **间接寻址开销**：Attention kernel 需要查 page table 做间接访存，比连续内存多一次 memory access。
4. **与 CUDA Graph 冲突**：CUDA Graph 要求固定形状，但 page table 是动态的 → 需要 capture 时预分配 buffer，replay 时 copy。

**Extension**：

- 如果要支持超长上下文（1M token），可以考虑 **层级 Page Table**（类似多级页表），减少单次查找的 table 大小
- 如果 page_size 很小（如 1），page table 查找开销大 → 可以用 **更大的 page_size**（Mini-SGLang 支持 16/32/64）来减少 table 条目数
- 如果要跨机推理，page table 需要支持 **远程内存访问**（RDMA）

**追问陷阱**：
- "page_size 设成多少最好？" → 取决于模型和场景。page_size 小 → 粒度细、浪费少，但 page table 大、kernel 效率低。page_size 大 → 对齐效率高但内部碎片多。TRT-LLM 后端强制 page_size=16/32/64。
- "为什么不直接用虚拟内存？" → CUDA 的虚拟内存管理（`cuMemMap`）粒度太粗（2MB），且管理开销大。

---

### Q2: Radix Cache 的设计思想是什么？为什么不用 hash table？

**Problem**：

多请求共享前缀（system prompt、few-shot、多轮对话历史）时，如果每个请求都独立计算前缀的 KV Cache，会浪费大量 GPU 计算。需要一个数据结构来**高效查找和复用共享前缀**。

最直觉的方案是 hash table：对每个 token 序列算 hash，直接查表。但 hash table 有几个问题：
1. 只能做**精确匹配**，不能做**前缀匹配**（"abc" 和 "abcd" 在 hash table 中是完全不同的 key）
2. 需要对每个可能的前缀长度都算 hash，O(n) 的预处理
3. 不支持增量更新（插入新前缀时需要重新算 hash）

**Design**：

Radix Tree（基数树）是一种压缩前缀树，天然支持前缀匹配。

Mini-SGLang 的实现（`kvcache/radix_cache.py:17-85`）：
- 每个节点存储一段 token 序列（key）和对应的物理位置（value）
- 匹配时从根节点逐层走，每层用 `get_match_len` 比较
- 部分匹配时执行 `split_at` 拆分节点
- 引用计数区分 evictable/protected
- LRU 淘汰：按 timestamp 排序的最小堆，驱逐最久未使用的叶子节点

```python
def _tree_walk(self, input_ids):
    node = self.root_node
    while prefix_len < indice_len:
        child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
        if child_node is None: return node, prefix_len
        match_len = node.get_match_len(input_ids[prefix_len:])
        match_len = align_down(match_len, self.page_size)
        if match_len != node.length:
            node = node.split_at(match_len)  # 部分匹配 → 拆分
            return node, prefix_len
        node.timestamp = tic  # 更新 LRU 时间戳
    return node, prefix_len
```

**Trade-off**：

1. **树的深度不可控**：如果每个 token 都分叉，树退化成链表，遍历 O(n)
2. **split 操作开销**：部分匹配时需要创建新节点、调整指针，O(1) 但有内存分配开销
3. **并发安全**：Mini-SGLang 是单线程 Scheduler，不需要锁。但生产环境需要考虑
4. **内存开销**：每个 RadixTreeNode 有 children dict、ref_count、timestamp 等元数据

对比 Nano-vLLM 的 hash chain 方案（`engine/block_manager.py:36-41`）：
- 每 256 token 一个 block，每个 block 算一个 hash
- `hash_to_block_id` 字典做全局查找
- 更简单但粒度更粗（256 token 对齐），不支持任意长度前缀匹配

**Extension**：

- **跨请求的 prefix sharing**：如果多个 Agent 共享同一个 system prompt，Radix Cache 可以跨请求复用
- **持久化 Cache**：将 Radix Tree 序列化到磁盘，重启后恢复
- **分布式 Cache**：多个推理实例共享同一个 Radix Cache（需要分布式 KV store）

**追问陷阱**：
- "Radix Cache 的命中率怎么衡量？" → `evictable_size / total_size`，命中率高说明前缀复用好
- "如果所有请求的 prompt 都完全不同，Radix Cache 还有用吗？" → 没用，退化成 NaiveCache。但实际场景中，Agent 的 system prompt 是高度共享的
- "LRU 在什么场景下不是好的淘汰策略？" → 如果有"热点前缀"被频繁访问，LRU 会把它淘汰掉。LFU 更适合这种场景

---

### Q3: Overlap Scheduling 解决什么问题？和流水线并行有什么关系？

**Problem**：

在推理的每个 step 中，CPU 需要做：
1. 接收新消息（ZMQ recv）
2. 调度下一个 batch（选择请求、分配 page table、构造 metadata）
3. 处理上一个 batch 的结果（检查 EOS、释放资源、发送结果）

这些 CPU 操作和 GPU 的 forward 计算是**串行**的。当 GPU 很快（如 H200）而 CPU 调度较慢时，CPU 成为瓶颈，GPU 在等 CPU。

**Design**：

Overlap Scheduling 的核心思想：让 CPU 的"准备下一个 batch"和 GPU 的"执行当前 batch"**并行**。

Mini-SGLang 的实现（`scheduler/scheduler.py:83-106`）用两个 CUDA Stream：
```python
def overlap_loop(self, last_data):
    # 1. CPU: 接收消息
    for msg in self.receive_msg(blocking=blocking):
        self._process_one_msg(msg)
    # 2. CPU: 准备下一个 batch（在 self.stream 上）
    forward_input = self._schedule_next_batch()
    # 3. GPU: 在 engine.stream 上启动 forward（异步）
    if forward_input is not None:
        with self.engine_stream_ctx:
            self.engine.stream.wait_stream(self.stream)
            ongoing_data = (forward_input, self._forward(forward_input))
    # 4. CPU: 处理上一个 batch 的结果（与 GPU forward 并行）
    self._process_last_data(last_data)
    return ongoing_data
```

关键：步骤 3 的 GPU forward 是异步的（在 engine.stream 上），步骤 4 的 CPU 处理在另一个 stream 上，两者并行。

**Trade-off**：

1. **代码复杂度**：overlap 模式需要维护两个 stream 的同步，`wait_stream` 如果写错会导致 race condition
2. **内存开销**：需要同时保留当前 batch 和上一个 batch 的数据（`ForwardData` tuple）
3. **只在 CPU-bound 时有效**：如果 GPU 计算本身很慢（如长序列 prefill），CPU 开销被自然隐藏，overlap 没有收益

**Extension**：

- 这和训练框架中的 **DataLoader prefetch** 是同一个思想
- 可以扩展为 **三级流水线**：CPU 准备 batch N+2，GPU 执行 batch N+1，CPU 处理 batch N 的结果
- 生产环境中，可以用 **异步 IO** 进一步减少 CPU 开销

**追问陷阱**：
- "怎么知道系统是 CPU-bound 还是 GPU-bound？" → 看 GPU 利用率（`nvidia-smi`），如果 GPU 利用率低（< 80%），大概率是 CPU-bound
- "Mini-SGLang 怎么做 ablation study？" → 设置环境变量 `MINISGL_DISABLE_OVERLAP_SCHEDULING=1`，对比有无 overlap 的吞吐差异

---

### Q4: CUDA Graph 的原理和限制是什么？

**Problem**：

Decode 阶段，batch size 通常 < 256，每步只生成 1 个 token，GPU 计算量很小，但 CPU 需要 launch 多个 CUDA kernel（attention、linear、norm 等），每个 kernel 的 launch 开销 ~5-10μs。如果模型有 32 层，每层 5 个 kernel，总 launch 开销 = 32 × 5 × 10μs = 1.6ms，而 GPU 实际计算可能只有 0.5ms。CPU launch 成为瓶颈。

**Design**：

CUDA Graph 将一系列 kernel 操作**录制**下来，形成一个计算图，之后直接 **replay**，避免逐个 kernel 的 CPU launch 开销。

Mini-SGLang 的实现（`engine/graph.py:105-144`）：
1. 预捕获多个 batch size 的 graph（`[1, 2, 4, 8, 16, 24, ...]`）
2. 用 dummy request 做 warmup + capture
3. 运行时选择最近的 bs，pad batch 到对应大小
4. 复制新数据到 buffer，replay graph

```python
def replay(self, batch):
    self.buffer.copy_from(batch)          # 复制新数据到预分配 buffer
    g = self.graph_map[batch.padded_size]  # 选择对应 bs 的 graph
    self.attn_backend.prepare_for_replay(batch)  # 更新 attention metadata
    g.replay()                            # 直接 replay，无需 CPU launch
    return self.buffer.logits[:batch.size]
```

**Trade-off**：

1. **只适合 decode**：prefill 的序列长度不固定，无法用固定形状的 graph
2. **显存开销**：每个 bs 的 graph 都需要独立的 buffer，且 capture 时需要做 warmup，占用额外显存
3. **Padding 浪费**：实际 bs=13 但需要 pad 到 bs=16，多算 3 个 dummy request
4. **不支持 dynamic control flow**：graph 中不能有 if/else，所有分支都必须执行

**Extension**：

- 可以用 **CUDA Graph with dynamic shapes**（较新的 CUDA 版本支持）来减少 padding 浪费
- 可以和 **torch.compile** 结合：torch.compile 做 kernel fusion，CUDA Graph 做 launch 优化
- 对于 MoE 模型，不同 expert 的选择是 dynamic control flow → 需要特殊处理（如 always dispatch to all experts with masking）

**追问陷阱**：
- "CUDA Graph 的 pool 是什么？" → `graph.pool()` 返回一个 CUDA 内存池，后续 graph 可以复用同一块内存，减少显存占用
- "如果 bs 超过了预捕获的最大 bs 怎么办？" → `can_use_cuda_graph` 返回 False，走 eager mode

---

### Q5: Tensor Parallelism 中 GQA 的处理方式有什么特殊之处？

**Problem**：

GQA（Grouped Query Attention）中，Q heads 和 KV heads 数量不同。例如 Llama-3-70B 有 64 Q heads 但只有 8 KV heads。做 TP 时，如果按 Q heads 切分（每 GPU 16 Q heads），KV heads 只有 8 个，2 GPU 时每个 GPU 分 4 个 KV heads，4 GPU 时每个 GPU 分 2 个。但如果 8 GPU 时，每个 GPU 只分到 1 个 KV head，GQA 的收益就降低了。

**Design**：

Mini-SGLang 的处理（`layers/linear.py:82-83`）：
```python
local_num_qo = div_even(num_qo_heads, tp_info.size)
local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
```

`allow_replicate=True` 意味着当 KV heads 不能被 TP size 整除时，允许**复制**。例如 8 KV heads，TP=3 → 每个 GPU 分到 3 heads（最后一个 GPU 多一个）。TP=8 时每个 GPU 正好 1 个 KV head。

**Trade-off**：

1. **Replicate 浪费显存**：如果 KV heads 被复制，每个 GPU 都存一份完整的 KV Cache
2. **All-reduce 开销**：TP 需要在每层做 all-reduce（O-Proj 和 MLP-Down），通信量和 hidden_size 成正比
3. **TP 度数受限**：TP size 不能超过 Q heads 数，否则每个 GPU 分不到完整的 head

**Extension**：

- 可以用 **Sequence Parallelism** 替代部分 TP 的 all-reduce，减少通信量
- 对于 MoE 模型，可以 **TP + EP（Expert Parallelism）** 结合
- 后训练中，如果用 DeepSpeed ZeRO，可以 **TP + DP + ZeRO** 三层并行

**追问陷阱**：
- "TP 和 DP 的区别？" → TP 切同一批数据的模型权重，DP 切不同数据。TP 通信频繁（每层），DP 只在梯度同步时通信
- "什么时候用 TP，什么时候用 DP？" → 模型放得下单卡就用 DP，放不下就用 TP。70B 以上通常需要 TP=4 或 TP=8

---

### Q6: FlashAttention 为什么快？和 FlashInfer 的区别是什么？

**Problem**：

标准 Attention 的实现需要将 Q×K^T 的完整矩阵写入 HBM（显存），然后再读回来做 softmax 和 V 的乘法。对于 seq_len=4096，这个中间矩阵大小 = 4096 × 4096 × 2bytes = 32MB，每次都要读写 HBM，带宽成为瓶颈。

**Design**：

FlashAttention 的核心思想：**不在 HBM 中存储完整的 attention 矩阵**，而是：
1. 将 Q/K/V 分成小块，每块放入 SRAM（片上缓存，比 HBM 快 10-100 倍）
2. 在 SRAM 中完成 attention 计算
3. 用 **online softmax** 技巧，分块计算 softmax 而不需要全局归一化
4. 直接在 SRAM 中累加结果，最后写回 HBM

FlashInfer 是专门为 **Paged KV Cache** 优化的 attention 实现：
- 原生支持 paged_kv_indices 做间接寻址
- 支持 batch prefill + batch decode 的混合调度
- 支持 CUDA Graph capture

Mini-SGLang 用 HybridBackend（`attention/base.py:37-63`）为 prefill 和 decode 选择不同的后端：
- Prefill：FlashAttention（变长序列处理更好）
- Decode：FlashInfer（paged KV cache 原生支持，batch decode 效率更高）

**Trade-off**：

1. **FlashAttention 不支持 page table**：需要额外的接口适配
2. **FlashInfer 的 plan 操作有开销**：每次 batch 变化都需要重新 plan（`_initialize_metadata_once`）
3. **FA3 在 Hopper 上很快，但在 Ampere 上不一定比 FA2 快**
4. **FlashInfer 的 fa3 后端反而比 fa2 慢**（代码注释：`backend="fa2"  # flashinfer fa3 is slow`）

**Extension**：

- **FA4** 支持 Blackwell 架构，Mini-SGLang 已预留接口
- **Ring Attention** 可以将长序列分到多 GPU 上做 attention，突破单卡显存限制
- 后训练中，如果做 long-context RLHF，FlashAttention 的 memory efficiency 至关重要

---

## 第二类：实验和方案验证能力

### 面试官在考什么

不是考你做了什么实验，而是考你：
- **怎么证明**某个优化是有效的（而不是"我觉得有效"）
- 实验的 **控制变量** 是什么
- 有没有做过 **ablation study**（消融实验）
- 遇到实验结果和预期不一致时怎么办

### 应答框架：假设 → 实验设计 → 结果 → 分析 → 结论

```
1. 假设：你认为某个优化会带来什么效果？
2. 实验设计：怎么控制变量？用什么指标衡量？
3. 结果：具体数据是什么？
4. 分析：结果是否符合预期？不符合的原因是什么？
5. 结论：这个优化值得上线吗？在什么条件下值得？
```

---

### Q7: 你怎么验证 Overlap Scheduling 的效果？

**假设**：Overlap Scheduling 可以通过隐藏 CPU 开销来提升 GPU 利用率，从而提高吞吐。

**实验设计**：

Mini-SGLang 提供了环境变量 `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` 来关闭 overlap，方便做 ablation study。

控制变量：
- 相同的模型（Qwen3-0.6B）
- 相同的硬件（1×H200）
- 相同的请求分布（256 sequences，input 100-1024 tokens，output 100-1024 tokens）
- 唯一变量：overlap scheduling 开/关

指标：
- 吞吐（tokens/s）
- GPU 利用率（`nvidia-smi dmon`）
- 每 step 的 CPU 时间 vs GPU 时间

**预期结果**：
- 小模型（Qwen3-0.6B）：GPU 计算快，CPU 调度成为瓶颈，overlap 效果明显
- 大模型（Qwen3-14B）：GPU 计算慢，CPU 开销被自然隐藏，overlap 效果不明显

**追问陷阱**：
- "如果 overlap 开了之后吞吐反而下降了，怎么排查？" → 检查是否有 stream 同步开销过大、是否有内存拷贝开销、是否有 batch 大小变化
- "怎么区分是 overlap 的效果还是 CUDA Graph 的效果？" → 分别开关两个功能做 2×2 实验

---

### Q8: Radix Cache 的命中率怎么衡量？命中率低怎么办？

**衡量方法**：

每次请求的 `match_prefix` 返回 `cached_len`，命中率 = `cached_len / input_len` 的平均值。

可以在 Scheduler 中加统计：
```python
total_input_len += input_len
total_cached_len += cached_len
hit_rate = total_cached_len / total_input_len
```

**命中率低的原因分析**：

1. **请求之间没有共享前缀** → Radix Cache 退化成 NaiveCache → 换 `--cache naive` 节省管理开销
2. **Cache 被频繁 evict** → `evictable_size` 太小 → 增大 `--num-pages` 或减小并发数
3. **前缀不完全匹配** → 比如 system prompt 中有动态变量（日期、用户 ID） → 前缀拆分策略
4. **page_size 太大** → 匹配必须 page 对齐 → 减小 `--page-size`

**追问陷阱**：
- "命中率高就一定好吗？" → 不一定。如果为了维护高命中率而拒绝 evict，可能导致新请求 OOM
- "Radix Cache 和 Prefix Cache 的区别？" → Radix Cache 是 SGLang 的实现，Prefix Cache 是 vLLM 的实现，核心思想相同但数据结构不同

---

### Q9: 怎么评估 Chunked Prefill 的 chunk_size 对性能的影响？

**实验设计**：

Mini-SGLang 通过 `--max-prefill-length` 控制 chunk_size。

| chunk_size | 效果 |
|------------|------|
| 很小（128） | 每个 chunk 很快，但 chunk 数多，调度开销大，prefill 延迟高 |
| 中等（2048） | 平衡点 |
| 很大（8192） | 接近不分块，峰值显存高，可能 OOM |
| max_seq_len | 等于不分块 |

**衡量指标**：
- 首 token 延迟（TTFT, Time To First Token）
- 峰值显存占用
- Decode 阶段的吞吐（chunked prefill 会占用 decode 的时间片）

**分析**：

Chunked prefill 的本质 trade-off 是 **prefill 延迟 vs decode 吞吐**：
- chunk 小 → prefill 更频繁地被 decode 请求打断 → decode 吞吐高，但 prefill 延迟高
- chunk 大 → prefill 一次性做完 → prefill 延迟低，但 decode 被阻塞

**追问陷阱**：
- "为什么文档说 chunk_size=128 不推荐？" → 太小的 chunk 导致调度 overhead 占比过大，且 attention kernel 的效率降低（小 batch）
- "chunk 边界处的 attention 怎么处理？" → 每个 chunk 独立做 extend attention，cu_seqlens 处理边界。前一个 chunk 的 KV Cache 已经写入 cache pool，后一个 chunk 可以 attend 到

---

## 第三类：问题定位能力

### 面试官在考什么

不是考你能不能描述问题，而是考你：
- 有没有 **系统化的排查思路**（而不是东一榔头西一棒子）
- 能不能 **缩小范围**（从现象到根因）
- 有没有 **监控和日志** 的习惯
- 遇到问题是 **自己查** 还是 **问别人**

### 应答框架：现象 → 假设 → 验证 → 根因 → 修复

```
1. 现象：具体表现是什么？（吞吐下降？延迟升高？OOM？）
2. 假设：可能的原因有哪些？（按可能性排序）
3. 验证：怎么验证每个假设？（日志？profile？ablation？）
4. 根因：最终确定的原因是什么？
5. 修复：怎么修的？有没有引入新问题？
```

---

### Q10: 推理服务上线后吞吐突然下降，怎么排查？

**Step 1: 确认是 GPU 问题还是 CPU/IO 问题**

```bash
# GPU 利用率和显存
nvidia-smi dmon -s u -d 1

# CPU 利用率
top -bn1 | head -20

# IO 等待
iostat -x 1
```

- GPU 利用率低（< 50%）→ CPU-bound（调度开销大、ZMQ 通信慢、tokenizer 慢）
- GPU 利用率高但吞吐低 → 可能是 batch size 小、或 kernel 效率低
- 显存不足 → OOM 导致请求被拒绝

**Step 2: 如果是 CPU-bound**

Mini-SGLang 有 NVTX annotation（`@nvtx_annotate`），可以用 Nsight Systems 做 profiling：
```bash
nsys profile -t cuda,nvtx -o trace python -m minisgl --model Qwen3-0.6B
```

看 NVTX 标记的各个阶段耗时：
- `MHA`（attention）→ 如果很短，说明 GPU 没充分利用
- `MLP`（feed-forward）→ 正常
- `Sampler` → 如果很长，可能是 FlashInfer sampling 的 PDL 问题

**Step 3: 如果是 kernel 效率低**

检查是否用了 CUDA Graph（decode 阶段）：
```bash
# 看日志中是否有 "Capturing CUDA graphs"
# 如果没有，说明 CUDA Graph 被禁用了
```

检查是否用了正确的 attention 后端：
```bash
# 看日志中 "Auto-selected attention backend"
# Hopper GPU 应该是 "fa,fi"（prefill 用 FA3，decode 用 FlashInfer）
```

**Step 4: 如果是显存不足**

Mini-SGLang 在初始化时会打印：
```
Free memory before loading model: XX GiB
Free memory after initialization: XX GiB
Allocating XXXX tokens for KV cache, K + V = XX GiB
```

如果 KV Cache 分配的 token 数太少 → 并发请求受限 → 吞吐下降。

**Step 5: 如果是 Radix Cache 污染**

如果某些请求的 prompt 很长但不共享前缀，Radix Cache 会被填满，导致频繁 evict → 新请求的 prefix cache 命中率下降 → prefill 计算增加。

**追问陷阱**：
- "如果 GPU 利用率很高但吞吐还是低呢？" → 可能是 batch size 太小（decode 阶段每个请求只生成 1 个 token），需要增大并发数
- "如果只是某些请求慢呢？" → 可能是长序列请求阻塞了短序列请求 → 检查 chunked prefill 是否生效

---

### Q11: 模型输出质量突然下降，怎么排查是推理框架的问题还是模型本身的问题？

**排查步骤**：

1. **用 greedy 模式复现**：`temperature=0`，看输出是否确定性。如果 greedy 输出正常但采样输出异常 → 采样逻辑问题
2. **对比离线推理**：用 Mini-SGLang 的 offline LLM 接口，同一个 prompt 直接调用，看输出是否一致
3. **检查 KV Cache**：如果输出越来越离谱 → 可能是 KV Cache 被污染（错误的 page table 映射）
4. **检查 tokenizer**：如果输出有乱码 → 可能是 tokenizer 和 detokenizer 不同步
5. **检查数值精度**：如果 bf16 推理有 NaN → 可能是某些 kernel 的数值不稳定

**Mini-SGLang 的防护措施**：
- `check_integrity()`（`scheduler/cache.py:81-91`）：检查 `free_pages + cache_pages == num_pages`
- `Req.__post_init__` 检查 `cached_len < device_len <= max_device_len`
- Page table 的 dummy request（`engine/engine.py:89-98`）：防止越界访问

**追问陷阱**：
- "如果推理框架没问题，但输出就是不好呢？" → 可能是量化问题、prompt template 问题、或模型本身的问题
- "怎么区分是 prefill 阶段的问题还是 decode 阶段的问题？" → 只生成 1 个 token，看 logits 分布是否正常

---

### Q12: 多卡 TP 部署时，某张卡的显存占用异常高，怎么排查？

**可能原因**：

1. **NCCL buffer 未释放**：TP 通信需要临时 buffer，如果通信异常可能未释放
2. **KV Cache 分配不均**：Mini-SGLang 在 `_sync_get_memory()` 中检查各卡显存差异（`engine/engine.py:182`），如果差异 > 2GB 会报错
3. **CUDA Graph buffer**：不同卡的 graph capture 进度不同，可能某些卡多分配了 buffer
4. **模型权重不均**：TP 切分时，如果某个 head 被分到多张卡，权重分配可能不均

**排查步骤**：
```bash
# 查看每张卡的显存分布
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

然后在代码中检查 `_sync_get_memory()` 的日志输出。

**Mini-SGLang 的防护**（`engine/engine.py:182-187`）：
```python
if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
    logger.error(f"Memory across TP ranks are imbalanced:")
    raise RuntimeError("Memory across TP ranks are imbalanced")
```

---

## 第四类：工程落地能力

### 面试官在考什么

不是考你知不知道理论，而是考你：
- 理论方案在**实际工程中**能不能跑通
- 有没有考虑 **稳定性、监控、回滚**
- 有没有处理 **边界情况** 的经验
- 能不能在 **资源有限** 的情况下做出取舍

### 应答框架：理论 → 工程挑战 → 解决方案 → 验证

```
1. 理论：这个方案理论上应该 work
2. 工程挑战：实际落地时遇到了什么问题？
3. 解决方案：怎么解决的？
4. 验证：怎么证明解决了？
```

---

### Q13: 推理框架怎么部署到生产环境？

**Mini-SGLang 的部署方式**：

```bash
# 单卡部署
python -m minisgl --model "Qwen/Qwen3-0.6B"

# 多卡 TP 部署
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000

# Docker 部署
docker build -t minisgl .
docker run --gpus all -p 1919:1919 minisgl --model Qwen/Qwen3-0.6B
```

**生产环境需要额外考虑**：

1. **健康检查**：需要一个 `/health` 端点，Kubernetes 可以定期探测
2. **优雅关闭**：收到 SIGTERM 后，完成正在处理的请求再退出。Mini-SGLang 用 `KeyboardInterrupt` 处理（`server/launch.py:33-37`）
3. **限流**：当并发请求超过 KV Cache 容量时，需要拒绝新请求或排队
4. **监控指标**：QPS、TTFT、TPOT（Time Per Output Token）、GPU 利用率、KV Cache 使用率
5. **日志**：Mini-SGLang 用 `init_logger` 做分级日志，生产环境需要接入 ELK
6. **回滚**：模型更新时，需要支持 A/B 测试或 canary 发布

**Mini-SGLang 的多进程架构对部署的影响**：
- 每个 TP rank 是一个独立进程 → 需要确保所有进程同时启动和关闭
- ZMQ IPC 地址用 PID 做后缀（`scheduler/config.py:8-11`）→ 避免多实例冲突
- Tokenizer 和 Detokenizer 是独立进程 → 可以独立扩缩容

**追问陷阱**：
- "如果 Scheduler 进程挂了怎么办？" → 需要 watchdog 机制，检测进程存活并自动重启
- "怎么做到零停机更新？" → 启动新实例 → 健康检查通过 → 流量切换 → 关闭旧实例

---

### Q14: 推理框架的内存管理怎么做？怎么避免 OOM？

**Mini-SGLang 的内存管理**：

1. **启动时预算**：`_determine_num_pages()` 计算可用显存，分配 KV Cache
   ```python
   model_memory = old_free_memory - new_free_memory
   available_memory = int(config.memory_ratio * old_free_memory) - model_memory
   num_pages = available_memory // cache_per_page
   ```

2. **运行时分配/回收**：`CacheManager` 管理 free_slots 池
   - 分配：从 free_slots 取页
   - 回收：还到 free_slots
   - Eviction：如果 free_slots 不够，从 Radix Cache evict

3. **完整性检查**：`check_integrity()` 验证 `free_pages + cache_pages == num_pages`

**可能的 OOM 场景**：

1. **请求的 output_len 估计不准**：如果用户设了 `max_tokens=100000`，但实际只生成了 100 个 token，预分配的 KV Cache 浪费了
2. **并发请求过多**：每个请求都需要 KV Cache 空间，超过总容量时需要 evict 或拒绝
3. **长序列 prefill**：chunked prefill 的 chunk 内部仍然需要临时显存

**防护措施**：

- Mini-SGLang 的 `PrefillAdder._try_allocate_one()` 在分配前检查容量（`scheduler/prefill.py:50-54`）
- 如果 `estimated_len + reserved_size > available_size`，拒绝加入 batch
- 二次检查：lock 之后再检查一次，防止并发分配导致超限

**追问陷阱**：
- "如果 evict 也腾不出空间怎么办？" → Mini-SGLang 会 assert 失败（`radix_cache.py:151-153`）。生产环境应该返回 HTTP 503，让客户端重试
- "memory_ratio 设多少合适？" → 默认 0.8-0.9，留 10-20% 给 CUDA kernel 的临时显存

---

### Q15: 怎么保证推理结果的可复现性？

**影响可复现性的因素**：

1. **采样随机性**：`temperature > 0` 时，输出不确定 → 设 `temperature=0` 做 greedy
2. **浮点精度**：bf16 的 reduce 顺序不同 → 结果可能有微小差异
3. **CUDA 的非确定性**：某些 CUDA kernel（如 `atomicAdd`）是不确定性的
4. **Batch 组成**：不同 batch 中的请求组合不同 → attention 计算的 padding 不同 → 结果可能有微小差异

**Mini-SGLang 的可复现性保证**：

```python
torch.manual_seed(42)  # engine/engine.py:37
```

但只控制了随机种子，没有控制 CUDA 的非确定性。如果需要严格可复现：
```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
```

**追问陷阱**：
- "为什么 bf16 的结果和 fp32 不一样？" → bf16 的精度只有 7 位有效数字，累积误差可能导致不同的 token 选择
- "如果要 debug 某个请求的输出，怎么做？" → 用 offline LLM 接口，只发这一个请求，greedy 模式，看 logits 分布

---

## 第五类：业务与实际场景的理解

### 面试官在考什么

不是考你技术多牛，而是考你：
- 知不知道这个技术 **在什么场景下有用**
- 能不能 **量化收益**（不只是"更快"，而是"从 100ms 降到 50ms"）
- 知不知道 **上线成本**（人力、机器、时间）
- 能不能在 **资源有限** 时做优先级排序

### 应答框架：场景 → 痛点 → 方案 → 收益 → 成本

```
1. 场景：这个技术用在什么业务场景？
2. 痛点：用户/业务的核心痛点是什么？
3. 方案：怎么解决的？
4. 收益：量化收益是什么？（延迟、吞吐、成本）
5. 成本：上线需要多少资源？
```

---

### Q16: Prefix Caching 在 Agent 场景中价值有多大？

**场景**：

Agent 调用 LLM 的典型模式：
1. System prompt + tool definitions（固定，~2000 tokens）
2. 用户问题（变化，~100 tokens）
3. LLM 返回 tool call（~200 tokens）
4. 执行 tool，将结果拼接到 prompt
5. 再次调用 LLM（prompt = system + user + tool_call + tool_result）

**痛点**：

每次 Agent 调用都需要重新计算 system prompt 的 KV Cache（~2000 tokens 的 prefill）。如果 Agent 一轮对话调用 5 次 LLM，其中 4 次是在重复计算 system prompt。

**方案**：

Prefix Caching 可以缓存 system prompt 的 KV Cache，后续调用直接复用：
- 第 1 次：完整 prefill 2000 + 100 = 2100 tokens
- 第 2-5 次：只 prefill 新增部分（tool_call + tool_result），~500 tokens
- 节省：4 × 2000 = 8000 tokens 的 prefill 计算

**量化收益**：

假设 prefill 速度 = 1000 tokens/s（Qwen3-14B, H200）：
- 无 prefix caching：5 次调用总 prefill = 2100 + 4×2600 = 12500 tokens → 12.5s
- 有 prefix caching：5 次调用总 prefill = 2100 + 4×600 = 4500 tokens → 4.5s
- **节省 64% 的 prefill 时间**

**成本**：

- Radix Cache 的内存开销：每个节点 ~100 bytes，1000 个并发请求 → ~100KB，可忽略
- KV Cache 的显存开销：2000 tokens × 40 layers × 8 kv_heads × 128 head_dim × 2 bytes = 160MB
- 实现复杂度：中等（Mini-SGLang 已有完整实现）

**优先级排序**：

如果资源有限，Prefix Caching 应该优先于以下优化：
1. ✅ Prefix Caching（高 ROI，实现成熟）
2. ✅ Chunked Prefill（防 OOM，实现简单）
3. ⚠️ Overlap Scheduling（收益取决于 CPU-bound 程度）
4. ⚠️ CUDA Graph（只对 decode 有效，需要额外显存）

**追问陷阱**：
- "如果 system prompt 每次都不一样呢？" → Prefix Caching 无效，退化成 NaiveCache。但实际场景中，Agent 的 system prompt 通常是固定的
- "多轮对话中，历史消息怎么处理？" → 历史消息是共享前缀的一部分，可以被 prefix caching 复用。但如果历史很长（> 4K tokens），需要考虑 eviction 策略

---

### Q17: 如果资源有限（只有 1 张 H200），应该优先优化推理系统的哪些部分？

**场景**：一个 Agent 产品，用 Qwen3-14B 做 backbone，日活 1000 用户，每个用户平均 10 轮对话，每轮 5 次 LLM 调用。

**分析**：

日请求量 = 1000 × 10 × 5 = 50000 次/天
每秒请求数 = 50000 / 86400 ≈ 0.58 QPS（峰值可能是 3-5 倍，约 2-3 QPS）

单卡 H200 的 Qwen3-14B 推理能力（continuous batching）：
- Prefill: ~2000 tokens/s
- Decode: ~500 tokens/s
- 平均每个请求：prefill 2000 tokens + decode 200 tokens → 耗时 1s + 0.4s = 1.4s
- 单卡最大并发：~10 个请求（受 KV Cache 显存限制）

**优先级排序**：

| 优先级 | 优化 | 原因 | 预期收益 |
|--------|------|------|----------|
| P0 | Prefix Caching | Agent 场景高复用，直接减少 60%+ prefill | TTFT 降低 60% |
| P0 | Continuous Batching | 基础能力，没有它无法并发 | 并发 1→10 |
| P1 | Chunked Prefill | 长 prompt 防 OOM | 可靠性提升 |
| P1 | Streaming | 用户体验，更早看到部分结果 | 感知延迟降低 50% |
| P2 | CUDA Graph | decode 加速 | 吞吐提升 20-30% |
| P3 | Overlap Scheduling | CPU-bound 时有效 | 吞吐提升 10-20% |
| P3 | TP | 单卡场景不需要 | N/A |

**关键洞察**：

1. **Prefix Caching 是 ROI 最高的优化**：Agent 场景的高前缀复用率让 prefix caching 的收益最大化
2. **Streaming 是用户体验的关键**：即使总延迟不变，streaming 让用户更早看到结果
3. **CUDA Graph 和 Overlap Scheduling 是锦上添花**：在单卡低并发场景下，收益有限
4. **TP 不需要**：Qwen3-14B 在单卡 H200 上放得下（bf16 约 28GB，H200 有 141GB）

**追问陷阱**：
- "如果并发量增加到 100 QPS 怎么办？" → 需要多实例部署（DP），每个实例独立服务。或者用更大的模型做 batch 推理
- "如果 prompt 很长（> 32K tokens）怎么办？" → 必须用 chunked prefill，且需要增大 KV Cache 的 page 数

---

### Q18: MoE 模型的推理和 Dense 模型有什么不同？在什么场景下应该用 MoE？

**MoE vs Dense 的推理差异**：

| 维度 | Dense (如 Qwen3-14B) | MoE (如 Qwen3-32B MoE) |
|------|----------------------|------------------------|
| 激活参数 | 14B 全部激活 | 32B 总参数，但每个 token 只激活 ~3B |
| 计算量 | 正常 | 每 token 计算量更少（只走 top-k experts） |
| 显存占用 | 14B × 2 bytes = 28GB | 32B × 2 bytes = 64GB（全部加载） |
| 通信量 | TP all-reduce | TP all-reduce + expert dispatch |
| 内存带宽 | 正常 | 需要加载所有 expert 权重（即使只用部分） |

**Mini-SGLang 的 MoE 实现**（`moe/fused.py`）：
1. `fused_topk`：用 sgl_kernel 做 top-k expert 选择
2. `moe_align_block_size`：将 token 按 expert 分组，对齐到 block_size
3. `fused_experts_impl`：用 Triton kernel 做 fused expert 计算

**关键 trade-off**：

MoE 的 **计算效率高**（每个 token 只算 top-k experts），但 **内存带宽要求高**（需要加载所有 expert 的权重到 SRAM）。在 decode 阶段（访存密集型），MoE 可能比 Dense 更慢，因为需要读取更多权重。

**什么场景用 MoE**：

1. **高吞吐、低延迟**：MoE 的每 token 计算量少，适合高并发场景
2. **模型质量要求高**：MoE 可以用更少的计算达到更好的效果（Scaling Law）
3. **显存充足**：MoE 的总参数量大，需要更多显存加载权重

**什么场景用 Dense**：

1. **低并发、长序列**：Dense 的内存带宽效率更高
2. **显存受限**：Dense 的显存占用更少
3. **TP 度数小**：MoE 的 expert dispatch 需要额外通信

**追问陷阱**：
- "MoE 的 load balancing 怎么做？" → 有些 token 可能被路由到同一个 expert，导致热点。可以用 auxiliary loss 来平衡
- "MoE 的 expert 可以动态加载吗？" → 理论上可以（offloading），但会增加延迟。Mini-SGLang 目前不支持

---

## 附录：面试中的"万能"追问应对

### "你遇到的最大挑战是什么？"

**回答模板**（基于阅读源码的理解）：

> "在理解 Mini-SGLang 的 overlap scheduling 时，我花了很多时间理解两个 CUDA stream 的同步机制。关键代码是 `self.engine.stream.wait_stream(self.stream)`——这行代码确保 engine stream 在执行 forward 之前，等待 scheduler stream 准备好 metadata。如果写反了，会导致 race condition。
>
> 我通过对比 overlap 和 non-overlap 模式的日志输出，逐步理解了数据流：overlap 模式下，`_process_last_data` 和 `_forward` 是并行执行的，而 non-overlap 模式下是串行的。这个理解让我认识到，CUDA 编程中 **stream 同步** 是最容易出错的地方。"

### "你从这个项目中学到了什么？"

**回答模板**：

> "三个最重要的收获：
>
> 1. **系统思维**：推理框架不是单点优化，而是 Prefill/Decode 调度、KV Cache 管理、Attention 后端、CUDA Graph、TP 通信等多个子系统的协同。单独优化一个子系统可能没有效果，甚至可能有负面影响。
>
> 2. **Trade-off 意识**：每个设计决策都有 trade-off。Page size 大 vs 小、Prefix caching 的 eviction 策略、Chunked prefill 的 chunk size——没有银弹，只有适合特定场景的最优解。
>
> 3. **工程细节的重要性**：比如 `pin_memory=True` + `non_blocking=True` 的 CPU→GPU 拷贝、`torch.cuda.Event` 做 stream 同步、`align_down` 做 page 对齐——这些细节决定了系统的最终性能。"

### "如果让你改进这个项目，你会怎么做？"

**回答模板**（按优先级排序）：

> "1. **Speculative Decoding 支持**：Agent 场景的 decode 延迟是用户体验的瓶颈，speculative decoding 可以减少 30-50% 的延迟。需要改 Scheduler（支持 draft model forward）、Engine（支持 verify step）、Sampler（支持 reject sampling）。
>
> 2. **Prefix Caching 持久化**：将 Radix Tree 序列化到磁盘，重启后恢复。对 Agent 场景的价值很大（system prompt 不用重新计算）。
>
> 3. **Constrained Decoding**：支持 token masking，让 Agent 的 function calling 输出合法的 JSON。需要在 Sampling 层增加 mask 逻辑。
>
> 4. **监控和可观测性**：增加 Prometheus metrics（QPS、TTFT、TPOT、GPU utilization、KV cache hit rate），方便生产环境监控。"

---

## 总结：五类能力的核心要点

| 能力 | 核心要点 | 一句话 |
|------|----------|--------|
| 底层原理 | Problem → Design → Trade-off → Extension | "我知道为什么这么做，也知道它的局限" |
| 实验验证 | 假设 → 实验设计 → 结果 → 分析 → 结论 | "我用数据证明有效，不是拍脑袋" |
| 问题定位 | 现象 → 假设 → 验证 → 根因 → 修复 | "我有系统化的排查思路" |
| 工程落地 | 理论 → 工程挑战 → 解决方案 → 验证 | "我知道理论和实际的差距" |
| 业务理解 | 场景 → 痛点 → 方案 → 收益 → 成本 | "我知道什么值得做，什么不值得" |
