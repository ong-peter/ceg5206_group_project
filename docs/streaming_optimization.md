# Streaming 单线程优化与瓶颈分析

> 历史记录：以下实验与命令描述的是 2026-10-17 的 C 优化，使用当时的独立比较配置。2026-11-08 已接入最新 A/B，统一构建和正式比较入口改为 `Makefile` / `bench.py`。当前命令见 [提交说明](member_c_submission.md)，当前验证见 [集成记录](integration_2026-11-08.md)。旧加速比不是最新 A/B/C 比较结果。

本次只修改本地工作副本。基准固定 Br=Bc=32、1 线程；没有通过增大 tile、改变 baseline 编译选项或放宽容差来获得加速。

## 为什么初版会稍慢

Streaming 删除了 N×N 分数矩阵，但仍需完成全部 QKᵀ 和乘 V 的计算。初版沿用普通逐分数点积，同时增加每个 key tile 的最大值、指数重缩放、运行状态更新和循环控制。原始小实验中 `(128,64)` 的 score buffer 只有 64 KiB，`(512,64)` 为 1 MiB；少分配这些内存并不保证 CPU 时间下降。这里没有测量 DRAM 流量或硬件 cache miss，不能把差异直接归因为 DRAM 瓶颈。

之前的测量也没有交错算法运行，几百分点的差异易受系统状态影响。新的交错实验确认：旧版在 d=64 的较大形状仍慢于 baseline，但在一些 d=32/128 形状也会稍快。因此，“streaming 必然比 baseline 慢”不成立。

两个证据帮助定位本次优化方向：

1. 临时插桩将旧版划分为分数计算和 online-update/PV 两段。在 `(512,64)` 和 `(2048,64)` 上，分数计算约占总时间的 59%。这是带计时开销的诊断数据，不替代主基准时间。
2. 汇编显示旧分数点积虽然包含向量乘法，仍需要按顺序做标量累加。单个分数的累加依赖限制了并行性。“已有 vectorized loop”不能说明点积已高效利用 SIMD。

此外，旧版对每个后续块都计算 alpha 并遍历 u 重缩放。随机输入下，最大值实际增加的比例只有 15.8%（512）和 6.0%（2048）。其余更新中 alpha 恰好等于 1，可安全省去这部分工作。

## 已实现的改动

### 小 K 块转置与八分数累加

在 worker scratch 中增加 `Bc×d` 个 float，保存当前 key tile 的转置。随后用 8 个独立累加器同时计算相邻的 8 个 key 分数。每个 feature 的一个 Q 值可以与 8 个连续 K 值相乘；各分数仍按 feature 顺序累加。Apple Clang 生成了两条四通道 `fmla.4s`，可在寄存器中同时推进这 8 个分数。

代码使用可移植 C++，没有 ARM 专属 intrinsic。其他 CPU 的向量化和性能需要在目标平台重新确认。剩余不足 8 个 key 使用普通点积，测试覆盖 Bc=7/8/9/31/32/33 和非整除 N/d。

每个查询块仍会重新打包其遍历到的 K tile；这部分成本完整计入 kernel 时间。没有缓存全量转置矩阵，也没有 persistent auxiliary array。

### 仅在最大值增加时重缩放

首块使用已清零的 l/u。后续若 `m_new == m`，alpha 为 1，保留旧 u 并加入新块；若 `m_new > m`，仍按原 online-softmax 递推同时重缩放 l/u。没有改变 softmax 语义、删除任何 key 或使用近似 exp。

## 同条件基准和消融

`scripts/benchmark_streaming.py` 在同一个进程内比较 5 个实现：

| 名称 | 内容 |
| --- | --- |
| baseline | 原 `attention.cpp`，未修改 |
| original | 冻结的优化前 streaming，实现保存在 `benchmarks/streaming_original.cpp` |
| rescale_only | 普通分数点积，仅跳过不必要的重缩放 |
| packed_only | K packing 和八分数累加，保留每块重缩放 |
| optimized | 两项优化均启用，即生产版本 |

所有版本使用同样的 `-O3 -std=c++17 -Wall -Wextra -Wpedantic`，不启用 fast-math 或 LTO。每个形状先用精确 FP32 输入生成独立 NumPy FP64 reference，验证所有实现后再计时。buffer 预分配；每版本 5 次预热、15 次采样，短调用批量执行；每轮轮换并反转算法次序。全部 packing、初始化和归一化包含在时间内。原始采样、误差、scratch 和环境均保存。`rescale_only` 为控制算术消融保留与优化版相同的 scratch 容量，但不使用其中的 K panel。

本机 Apple M4 Pro、macOS 26.6.2、Apple Clang 17.0.0；没有固定 CPU affinity 或频率。12 个主形状均通过基准内的独立 FP64 校验。结果是本机 warm-call 测量，不能推广为其他 CPU 的固定加速比。

| N, d | Baseline ms | 旧 streaming ms | 优化 streaming ms | 相对旧版加速 | 相对 baseline 加速 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128, 64 | 0.264 | 0.262 | 0.168 | 1.56× | 1.57× |
| 512, 64 | 4.018 | 4.227 | 2.696 | 1.57× | 1.49× |
| 2048, 64 | 65.067 | 68.165 | 42.845 | 1.59× | 1.52× |
| 4096, 64 | 266.740 | 277.198 | 173.019 | 1.60× | 1.54× |
| 4096, 128 | 608.515 | 595.524 | 311.583 | 1.91× | 1.95× |

全部 12 个形状相对旧版加速 1.31–1.91×，相对 baseline 加速 1.35–1.95×。这些时间不与上一轮单独运行的时间直接拼接计算 speedup。

在 `(2048,64)` 上，消融结果为：

| 版本 | 中位数 ms |
| --- | ---: |
| original | 68.165 |
| rescale_only | 64.784 |
| packed_only | 43.721 |
| optimized | 42.845 |

主要收益来自分数计算的 SIMD 排列。只跳过重缩放在此例约减少 5% 时间，不足以解释完整加速。阶段插桩也观察到 score/packing 段从约 41.46 ms 降至 18.98 ms；优化后 online-update/PV 已成为较大的剩余部分。插桩会改变编译和计时开销，阶段数据只用于定位，不能与未插桩总时间直接相加。

## 内存和正确性约束

令 `R=min(Br,N)`、`C=min(Bc,N)`、`W=min(threads,ceil(N/R))`：

```text
scratch bytes = 4W × [RC + Rd + 2R + Cd + (W>1 ? 32 : 0)]
Q/K/V/O bytes = 16Nd
persistent auxiliary = 0
```

相对旧版增加的只有每 worker 的 C×d 个 float。`N=4096,d=64,Br=Bc=32` 时：

| 实现 | Scratch |
| --- | ---: |
| Baseline | 64 MiB |
| 旧 streaming，1 线程 | 12.25 KiB |
| 优化 streaming，1 线程 | 20.25 KiB |
| 优化 streaming，4 线程容量 | 81.5 KiB |

所有内存查询与测试断言已同步加入 K panel。scratch 预分配，不在 kernel 内申请内存；同一查询的完整 online recurrence 仍由一个线程负责。多 worker 的 packing buffer 也彼此独立。

容差保持 `abs(O-Oref) <= 1e-4 + 1e-3*abs(Oref)`。SIMD/FMA 可能改变最后几位，验证目标是满足该误差契约，不要求与旧版逐位一致。相同实现、tile 与输入在不同线程数之间仍需逐位一致。

单线程和 4 线程各 111 项完整参考检查均通过，最大绝对误差为 `1.1178691438741908e-5`，最大相对 Frobenius 误差为 `1.458256254275443e-6`。这与旧版误差不完全相同，但没有放宽任何容差。201 组单线程 API 检查、601 组多线程 API 检查及 UBSan 均通过。

本阶段的测试记录在 `results/optimization/`。

- `correctness_serial.csv`：baseline 与 streaming 的全部 12 个形状、边界和特殊输入。
- `correctness_openmp4.csv`：4 线程 streaming 的完整参考检查。
- `api_serial.log`、`openmp_build.log`：201 / 601 组 API 配置，含新 SIMD 尾部边界、输入保护、NaN scratch、缓冲区复用和线程一致性。
- `ubsan.log` 记录本阶段的 UBSan 检查。后续使用 LLVM 22.1.8 完成的 ASan 复测见[集成记录](integration_2026-11-08.md)。
- `memory_serial.csv`、`memory_openmp4.csv`：更新后的内存统计。
- `comparison.csv`：12 个形状、5 个实现的未插桩交错测量。
- `profile.csv`：单独运行的阶段计时和最大值变化计数。
- `streaming.s`、`streaming_original.s`、`vectorization.txt`：汇编和向量化证据。

## 运行和依赖

在仓库根目录运行下列命令，默认编译优化版。

```sh
bash scripts/run_member_c.sh
```

复现本次比较（默认 3 个代表形状，`--full` 为 12 个）：

```sh
.venv/bin/python scripts/benchmark_streaming.py
.venv/bin/python scripts/benchmark_streaming.py --full
.venv/bin/python scripts/profile_streaming.py
```

使用 `--output results/optimization/another_run` 可保留其他比较结果；默认会覆盖同名 CSV。profile 只生成并编译临时副本，不修改生产源文件。

**C 不依赖 B 或 D 尚未完成的代码。** 生产 kernel 只需要 `attention.h`、`streaming.cpp` 和 C++17 标准库；多线程另需 OpenMP。现有 CLI、baseline、FP64 reference 来自仓库已有基础。`benchmarks/` 的旧版本只服务于前后对照，不参与普通 `make`。

B 仍负责独立的 blocked-materialized 实现，之后可与 C 协调共享 microkernel；这次局部 K packing 是让 C 独立运行的实现选择，没有实现或覆盖 B 的算法家族。D 可直接使用现有线程接口，继续完成正式调优、绑核、线程扩展和硬件 profiling。本次诊断脚本不要求等待 D，也不等于团队全部实验已经完成。
