# Mini-SGLang 8×RTX 3090 复现计划

> 基于实际硬件环境（8×RTX 3090 24GB, SM86, PCIe, CUDA 13.1, Driver 590.48.01）制定的完整复现方案。

---

## 一、硬件与环境评估

### 1.1 硬件规格

| 项目 | 规格 | 对推理框架的影响 |
|------|------|-----------------|
| GPU | 8× RTX 3090, 24GB VRAM | 单卡可放 0.6B-8B 模型，大模型需 TP |
| 架构 | SM86 (Ampere) | **不支持 FA3/FA4/TRT-LLM**，只能用 FlashInfer(fa2) |
| 互联 | PCIe（无 NVLink） | TP 通信带宽受限，TP≤4 为宜 |
| CUDA | 13.1 | 非常新，PyTorch 兼容性需验证 |
| 总显存 | 192GB | 可放 32B bf16 模型（需 TP4） |

### 1.2 SM86 架构兼容性分析（关键）

代码中的架构检查点（`utils/arch.py`）：

```python
def is_sm90_supported() -> bool:   return is_arch_supported(9, 0)   # 3090 = SM86 → False
def is_sm100_supported() -> bool:  return is_arch_supported(10, 0)  # → False
```

Attention 后端自动选择逻辑（`engine/engine.py:222-225`）：
```python
if config.attention_backend == "auto":
    backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
    # SM86 → 自动选择 "fi"（FlashInfer only）
```

**各后端在 SM86 上的兼容性**：

| 后端 | SM86 支持 | 原因 |
|------|----------|------|
| FlashInfer (fi) | ✅ 支持 | FlashInfer 的 fa2 后端支持 SM80+ |
| FlashAttention (fa) | ❌ 不支持 | sgl_kernel 的 flash_attn_with_kvcache 需要 SM90+ |
| TRT-LLM | ❌ 不支持 | 需要 SM100+ |
| HybridBackend (fa,fi) | ❌ 不支持 | 因为 fa 不支持 SM86 |

**结论**：3090 上只能用 `--attn fi`（纯 FlashInfer），不能用 FA 和 TRT-LLM。

### 1.3 CUDA 13.1 兼容性风险

CUDA 13.1 是 2025/2026 的新版本。主要风险：
- PyTorch 官方 wheel 通常只编译到 CUDA 12.x → 可能需要从源码编译或用 nightly 版本
- sgl_kernel 的预编译 wheel 可能不包含 CUDA 13.1 版本
- tvm_ffi 的 JIT 编译依赖 nvcc 13.1，理论上可以工作但未充分测试

**降级方案**：如果 CUDA 13.1 兼容性问题太多，使用 Docker 容器（CUDA 12.8）。

---

## 二、模型选择与显存预算

### 2.1 各模型显存需求

| 模型 | 参数量 | bf16 显存 | 最少 GPU 数 | TP 配置 | 并发能力 |
|------|--------|----------|------------|---------|---------|
| Qwen3-0.6B | 0.6B | ~1.2GB | 1 | TP=1 | 高（~50 并发） |
| Qwen3-1.7B | 1.7B | ~3.4GB | 1 | TP=1 | 高 |
| Qwen3-4B | 4B | ~8GB | 1 | TP=1 | 中 |
| Qwen3-8B | 8B | ~16GB | 1 | TP=1 | 低（显存紧张） |
| Qwen3-14B | 14B | ~28GB | 2 | TP=2 | 中 |
| Qwen3-32B | 32B | ~64GB | 4 | TP=4 | 低 |
| Qwen3-30B-A3B (MoE) | 30B总/3B激活 | ~60GB | 4 | TP=4 | 中 |

**KV Cache 额外开销**（以 page_size=1, max_seq_len=4096 为例）：

```
单条 KV Cache = 2 × num_layers × num_kv_heads × head_dim × seq_len × 2bytes
Qwen3-0.6B: 2 × 28 × 4 × 64 × 4096 × 2 = 112MB / 请求
Qwen3-14B:  2 × 40 × 8 × 128 × 4096 × 2 = 640MB / 请求
```

### 2.2 推荐复现路径

按从易到难的顺序：

| 阶段 | 模型 | GPU 配置 | 目的 |
|------|------|---------|------|
| Phase 1 | Qwen3-0.6B | 1×3090 | 跑通全流程，理解代码 |
| Phase 2 | Qwen3-14B | 2×3090 (TP=2) | 验证 TP + 多卡通信 |
| Phase 3 | Qwen3-32B | 4×3090 (TP=4) | 大模型 + 性能基准测试 |
| Phase 4 | 对比实验 | 1×3090 | Ablation: radix vs naive, overlap on/off |

---

## 三、环境搭建步骤

### 3.1 创建虚拟环境

```bash
cd /home/chenyizhou/mini-sglang

# 用 uv 创建 venv（比 conda 快）
uv venv --python=3.12
source .venv/bin/activate

# 确认 CUDA toolkit 在 PATH 中
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 3.2 安装 PyTorch（CUDA 13.1 兼容）

```bash
# 方案 A：尝试 PyTorch nightly（最可能支持 CUDA 13.1）
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu131

# 方案 B：如果方案 A 失败，用 CUDA 12.8 版本（向前兼容）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 方案 C：如果都不行，用 Docker（见 3.5）
```

验证：
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
# 期望输出: 2.x.x 13.1 (8, 6)  或  2.x.x 12.8 (8, 6)
```

### 3.3 安装核心依赖

```bash
# FlashInfer（SM86 支持的 fa2 后端）
uv pip install flashinfer-python>=0.5.3

# sgl_kernel（注意：可能需要从源码编译以支持 CUDA 13.1）
uv pip install sgl-kernel>=0.3.17.post1
# 如果预编译 wheel 不支持当前 CUDA/Python 版本，从源码安装：
# uv pip install "sgl-kernel>=0.3.17.post1" --no-binary sgl-kernel

# TVM FFI（JIT 编译后端）
uv pip install apache-tvm-ffi>=0.1.4

# 其他依赖
uv pip install accelerate msgpack modelscope pyzmq uvicorn fastapi prompt_toolkit openai quack-kernels

# transformers 版本要求严格
uv pip install "transformers>=4.56.0,<=4.57.3"
```

### 3.4 安装 mini-sglang 本身

```bash
cd /home/chenyizhou/mini-sglang
uv pip install -e .
```

### 3.5 Docker 降级方案（如果 CUDA 13.1 兼容性问题太多）

```bash
# 使用项目自带的 Dockerfile（CUDA 12.8）
docker build -t minisgl .

# 运行（需要 NVIDIA Container Toolkit）
docker run --gpus all -p 1919:1919 \
    -v /home/chenyizhou/.cache/huggingface:/app/.cache/huggingface \
    minisgl --model "Qwen/Qwen3-0.6B" --host 0.0.0.0
```

### 3.6 安装验证脚本

```bash
# 验证脚本：确认所有关键组件可以加载
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Capability: {torch.cuda.get_device_capability(0)}')

import flashinfer
print(f'FlashInfer: {flashinfer.__version__}')

import tvm_ffi
print(f'TVM FFI: OK')

from minisgl.utils import is_sm90_supported, is_sm100_supported
print(f'SM90: {is_sm90_supported()}')   # 应该是 False
print(f'SM100: {is_sm100_supported()}')  # 应该是 False

print('All checks passed!')
"
```

---

## 四、分阶段复现方案

### Phase 1：单卡跑通 Qwen3-0.6B（Day 1-2）

**目标**：验证安装正确性，跑通完整推理流程。

#### 1.1 下载模型

```bash
# 从 HuggingFace（如果网络好）
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')"

# 或从 ModelScope（国内更快）
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --dummy-weight
# --dummy-weight 先不下载真实权重，验证框架能否启动
```

#### 1.2 Dummy Weight 快速验证

```bash
# 用 dummy weight 验证框架启动和推理流程
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --dummy-weight
```

预期输出：
```
Auto-selected attention backend: fi
Free memory before loading model: XX GiB
Free memory after initialization: XX GiB
Allocating XXXX tokens for KV cache
API server is ready to serve on 127.0.0.1:1919
```

#### 1.3 真实权重推理

```bash
# Shell 模式（最简单的交互方式）
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --shell

# 服务模式
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --port 1919

# 另一个终端测试
curl http://localhost:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}'
```

#### 1.4 Offline Benchmark

```bash
# 修改 benchmark 脚本中的模型路径
python benchmark/offline/bench.py
```

#### 1.5 学习检查点

- [ ] 能解释 `--attn fi` 为什么被自动选择（SM86 → 只有 FlashInfer）
- [ ] 能画出请求从 API Server 到 Tokenizer 到 Scheduler 到 Engine 的完整数据流
- [ ] 能解释 KV Cache 的显存是怎么计算和分配的（`_determine_num_pages`）
- [ ] 能解释 Radix Cache 的 tree_walk 过程

---

### Phase 2：多卡 TP=2 跑 Qwen3-14B（Day 3-5）

**目标**：验证 Tensor Parallelism，理解多卡通信。

#### 2.1 启动 TP=2

```bash
# Qwen3-14B 需要 ~28GB 显存，单卡 24GB 放不下
# TP=2，每张卡放一半权重
CUDA_VISIBLE_DEVICES=0,1 python -m minisgl \
    --model "Qwen/Qwen3-14B" \
    --model-source modelscope \
    --tp 2 \
    --port 1919
```

#### 2.2 验证 TP 通信

```bash
# 发送请求，观察两卡的 GPU 利用率
nvidia-smi dmon -s u -d 1 &
curl http://localhost:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-14B","messages":[{"role":"user","content":"Explain quantum computing in 3 sentences."}],"max_tokens":200}'
```

#### 2.3 PCIe 带宽瓶颈观察

```bash
# 对比 TP=1 和 TP=2 的吞吐差异
# PCIe 带宽 ~32GB/s per direction (x16 Gen4)
# NVLink 带宽 ~600GB/s
# 预期：TP=2 的 per-GPU 吞吐不是 TP=1 的 2 倍，而是 ~1.3-1.5 倍（受 PCIe 限制）
```

#### 2.4 学习检查点

- [ ] 能解释 TP 中 Column Parallel 和 Row Parallel 的区别
- [ ] 能解释为什么 QKV Projection 是 Column Parallel，Output Projection 是 Row Parallel
- [ ] 能解释 all-reduce 在哪些地方发生、通信量多大
- [ ] 能解释 GQA 中 `allow_replicate=True` 的含义

---

### Phase 3：大模型 TP=4 跑 Qwen3-32B（Day 6-8）

**目标**：大模型推理，理解显存管理和调度策略。

#### 3.1 启动 TP=4

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m minisgl \
    --model "Qwen/Qwen3-32B" \
    --model-source modelscope \
    --tp 4 \
    --port 1919 \
    --memory-ratio 0.85
```

#### 3.2 并发压力测试

```bash
# 用 benchmark 脚本测试并发能力
python benchmark/offline/bench.py
```

#### 3.3 显存监控

```bash
# 实时监控各卡显存
watch -n 1 nvidia-smi

# 或用更详细的工具
nvidia-smi dmon -s pucvmet -d 1
```

#### 3.4 学习检查点

- [ ] 能解释 `_determine_num_pages` 的内存预算逻辑
- [ ] 能解释 `_sync_get_memory` 为什么要做 all-reduce 取 min/max
- [ ] 能解释 `memory_ratio` 参数的作用和调优方法
- [ ] 能解释 Chunked Prefill 的 `token_budget` 如何影响显存和延迟

---

### Phase 4：Ablation 对比实验（Day 9-12）

**目标**：通过控制变量实验，深入理解每个优化组件的贡献。

#### 4.1 实验 1：Radix Cache vs Naive Cache

```bash
# 有 Radix Cache
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --cache radix --port 1919

# 无 Radix Cache（Naive）
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --cache naive --port 1920

# 用相同的请求序列测试，对比吞吐
# 预期：如果有共享前缀，radix 比 naive 快 30-60%
```

#### 4.2 实验 2：Overlap Scheduling on/off

```bash
# Overlap ON（默认）
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --port 1919

# Overlap OFF
MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --port 1920

# 对比吞吐差异
# 预期：小模型（CPU-bound）差异明显，大模型差异小
```

#### 4.3 实验 3：CUDA Graph on/off

```bash
# CUDA Graph ON（默认，自动选择 bs）
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --port 1919

# CUDA Graph OFF
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --cuda-graph-max-bs 0 --port 1920

# 对比 decode 吞吐
# 预期：CUDA Graph 可以提升 decode 吞吐 20-40%
```

#### 4.4 实验 4：Page Size 影响

```bash
# page_size=1（最细粒度）
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --page-size 1 --port 1919

# page_size=16
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --page-size 16 --port 1920

# page_size=256
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --page-size 256 --port 1921

# 对比：显存利用率、吞吐、Radix Cache 命中率
```

#### 4.5 实验 5：Chunked Prefill chunk_size 影响

```bash
# chunk_size=2048（默认）
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --max-prefill-length 2048 --port 1919

# chunk_size=512
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --max-prefill-length 512 --port 1920

# chunk_size=8192
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --max-prefill-length 8192 --port 1921

# 对比：首 token 延迟（TTFT）、峰值显存
```

#### 4.6 实验 6：MoE 模型（如果时间允许）

```bash
# Qwen3-30B-A3B 是 MoE 模型，30B 总参数但每 token 只激活 3B
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m minisgl \
    --model "Qwen/Qwen3-30B-A3B" \
    --model-source modelscope \
    --tp 4 \
    --port 1919

# 对比 MoE vs Dense 的吞吐差异
# 预期：MoE 的每 token 计算量少，decode 吞吐更高
```

---

## 五、代码走读计划

### 5.1 优先级排序

按**从请求进来到响应出去**的顺序读代码：

| 优先级 | 模块 | 文件 | 行数 | 核心问题 |
|--------|------|------|------|----------|
| P0 | 核心数据结构 | `core.py` | 136 | Req/Batch/Context 是什么？ |
| P0 | Scheduler 主循环 | `scheduler/scheduler.py` | 267 | 请求怎么被调度的？ |
| P0 | Engine | `engine/engine.py` | 233 | 模型怎么 forward 的？ |
| P1 | Prefill 管理 | `scheduler/prefill.py` | 162 | Chunked Prefill 怎么实现？ |
| P1 | Cache 管理 | `scheduler/cache.py` | 146 | Page 怎么分配和回收？ |
| P1 | Radix Cache | `kvcache/radix_cache.py` | 237 | 前缀缓存怎么工作？ |
| P2 | Attention 后端 | `attention/fi.py` | 271 | FlashInfer 怎么调用？ |
| P2 | CUDA Graph | `engine/graph.py` | 171 | Graph 怎么 capture/replay？ |
| P2 | TP Linear | `layers/linear.py` | 127 | 权重怎么切分？ |
| P3 | KV Cache 物理存储 | `kvcache/mha_pool.py` | 68 | 物理存储形状是什么？ |
| P3 | Sampler | `engine/sample.py` | 75 | 采样怎么做？ |
| P3 | MoE | `moe/fused.py` | 256 | MoE kernel 怎么实现？ |

### 5.2 阅读方法

1. **先跑通再读**：Phase 1 跑通后再开始读代码
2. **断点调试**：在关键函数（`_schedule_next_batch`, `forward_batch`, `match_prefix`）打断点
3. **加日志**：在关键路径加 print，观察运行时状态
4. **画图**：画出数据流图和类关系图

---

## 六、可能遇到的问题与解决方案

### 6.1 PyTorch + CUDA 13.1 兼容性

**问题**：PyTorch 预编译 wheel 可能不支持 CUDA 13.1

**解决方案**：
1. 尝试 PyTorch nightly 版本
2. 尝试用 CUDA 12.8 版本的 PyTorch（CUDA 13.1 驱动向下兼容 12.x runtime）
3. 用 Docker 容器（自带 CUDA 12.8）

### 6.2 sgl_kernel 安装失败

**问题**：sgl_kernel 预编译 wheel 可能不支持 Python 3.12 + CUDA 13.1

**解决方案**：
1. `uv pip install sgl-kernel --no-binary sgl-kernel`（从源码编译）
2. 检查 sgl_kernel 的 `setup.py` 是否有 SM86 的 arch flag
3. 如果不行，修改 `setup.py` 添加 `sm_86`

### 6.3 FlashInfer 在 SM86 上效率低

**问题**：FlashInfer 的 tensor core 利用率在 SM86 上可能不如 SM90

**解决方案**：
```bash
# 关闭 tensor core 模式（如果 GQA ratio < 4）
MINISGL_FLASHINFER_USE_TENSOR_CORES=0 python -m minisgl --model ...
```

### 6.4 TP=4 时 PCIe 带宽瓶颈

**问题**：3090 没有 NVLink，PCIe 带宽 ~32GB/s，TP=4 时 all-reduce 成为瓶颈

**解决方案**：
1. 优先用 TP=2（只跨一条 PCIe 链路）
2. 如果必须 TP=4，监控 GPU 利用率，如果 < 60% 说明通信是瓶颈
3. 考虑用 `--disable-pynccl` 切换到 gloo 后端（但通常更慢）

### 6.5 显存不足 OOM

**问题**：3090 只有 24GB，大模型 + KV Cache 可能 OOM

**解决方案**：
1. 减小 `--memory-ratio`（默认 0.9，可以试 0.8）
2. 减小 `--max-running-requests`（减少并发 KV Cache）
3. 增大 `--page-size`（减少 page table 开销）
4. 使用 `--max-prefill-length` 限制 chunk 大小

---

## 七、时间线总览

| 天数 | 任务 | 产出 |
|------|------|------|
| Day 1 | 环境搭建 + 安装验证 | 能 `import minisgl` 不报错 |
| Day 2 | Phase 1: 单卡 Qwen3-0.6B 跑通 | Shell 模式能对话 + benchmark 数据 |
| Day 3-4 | Phase 2: TP=2 Qwen3-14B | 多卡推理成功 + 理解 TP 原理 |
| Day 5 | 代码走读 P0 模块 | 能画出完整数据流图 |
| Day 6-7 | Phase 3: TP=4 Qwen3-32B | 大模型推理 + 显存管理理解 |
| Day 8-9 | Phase 4: Ablation 实验 1-3 | radix/overlap/graph 对比数据 |
| Day 10-11 | Phase 4: Ablation 实验 4-6 | page_size/chunk_size/MoE 对比数据 |
| Day 12 | 代码走读 P2-P3 模块 | 能解释所有核心模块的实现 |
| Day 13-14 | 总结 + 面试准备 | 完善面试回答 |

---

## 八、命令速查

```bash
# ===== 环境 =====
cd /home/chenyizhou/mini-sglang
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda

# ===== 单卡 =====
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --shell
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --port 1919

# ===== 多卡 TP =====
CUDA_VISIBLE_DEVICES=0,1 python -m minisgl --model "Qwen/Qwen3-14B" --model-source modelscope --tp 2
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m minisgl --model "Qwen/Qwen3-32B" --model-source modelscope --tp 4

# ===== Ablation =====
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --cache naive
MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --cuda-graph-max-bs 0
python -m minisgl --model "Qwen/Qwen3-0.6B" --model-source modelscope --page-size 16

# ===== Benchmark =====
python benchmark/offline/bench.py

# ===== 测试 =====
cd /home/chenyizhou/mini-sglang && python -m pytest tests/ -v

# ===== 监控 =====
nvidia-smi dmon -s u -d 1
watch -n 1 nvidia-smi
```
