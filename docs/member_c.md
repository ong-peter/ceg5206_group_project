# C 的实现与统一运行方式

本地已集成远端 `855179728f95b755974f7a1ee5ecc6c462bb8771`（2026-11-07），包含 A、B 以及最新 benchmark/threading 基础。当前交付和验证入口分别是 [提交说明](member_c_submission.md) 与 [集成记录](integration_2026-11-08.md)。[10 月优化记录](streaming_optimization.md) 保留为历史实验，不作为当前 A/B/C 加速结论。

## 计算与实现

三种算法均计算 `O = softmax(QKᵀ / sqrt(d)) V`：FP32、row-major、单头、dense、无 mask。A 的 baseline 和独立 FP64 reference 保留；B 的 blocked 保留完整 N×N score；C 按 Br 查询行、Bc key/value 行流式处理。

C 每行保留最大值 m、归一化和 l、加权向量 u。首块使用清零的 l/u；后续最大值增加时，用 `alpha=exp(m-m_new)` 同时缩放 l/u。最大值不变时 alpha=1，跳过多余的重缩放。全部 key 处理完才输出 u/l。

C 将当前 K tile 转置到 worker scratch，通过八个独立分数累加器支持 SIMD，剩余不足八个 key 使用普通点积。packing、初始化、指数计算、归一化和线程开销均在 kernel call 内。没有近似 exp、fast-math、跨调用 K 缓存或 kernel 内动态分配。FMA 可能改变最后几位，接受规则保持 `abs(O-Oref) <= 1e-4 + 1e-3*abs(Oref)`。

## 内存契约

令 R=min(Br,N)、C=min(Bc,N)、W=min(threads,ceil(N/R))。

| 项目 | 每 worker 的 float 数量 |
| --- | ---: |
| score / 指数权重共用 tile | RC |
| 加权向量 u | Rd |
| m 和 l | 2R |
| 当前转置 K tile | Cd |
| 多 worker 间隔（W>1） | 32 |

```text
streaming scratch bytes = 4W × (RC + Rd + 2R + Cd + (W>1 ? 32 : 0))
baseline / current blocked scratch bytes = 4N²
Q/K/V/O bytes = 16Nd
persistent auxiliary arrays = 0
```

查询函数返回 float 数量，runner 再换算字节。统计包含预分配容量、尾块未用空间和 worker 间隔，不包含线程栈、运行库或验证程序的内存，也不是实测 DRAM 流量。固定 tile、d 和线程数时，C scratch 不随 N 二次增长。

`N=4096,d=64,Br=Bc=32` 时：A/B scratch 为 64 MiB；C 单线程 20.25 KiB，4 线程容量 81.5 KiB；Q/K/V/O 共 4 MiB。`scripts/memory_report.py` 现在要求三种算法都已编译，核对 12 形状共 36 行，而非旧版仅 A/C 的 24 行。

## 线程与 SIMD

每个查询块由一个 worker 完整处理，整行 recurrence 不跨 worker 拆分。Q/K/V 共享只读，输出行和全部 scratch 独立。scratch worker 间隔为 128 B；很小的输出行仍可能共享 cache line，未修改公共 O 布局。静态调度不保证小 N 或少量 query block 下充分利用所有线程。

当前使用统一 Makefile 的 ISA/SIMD 编译设置。C 不使用 B 的私有函数，也不强制改成相同 microkernel；无向量化构建禁用自动向量化/SLP，C 没有绕过该设置的硬编码 SIMD intrinsic。B 的 reduction 与 C 的逐 feature 累加方式不同，所以算法家族间的时间差不能全部归因于 online softmax。纯 C 优化消融仍可用历史诊断脚本单独研究。

## 运行入口

在仓库根目录、有 `.venv` 时：

```sh
bash scripts/run_member_c.sh
```

脚本执行：统一构建 vec/novec → C API 测试 → 三算法参考测试 → `bench.py` 小型比较 → 三算法内存报告。`check.py --bench` 已移除，正确性与时间分别保存为 `results/member_c_quick.csv` 和 `results/member_c_bench.csv`。

当前 Mac 的多线程入口：

```sh
OPENMP=1 LIBOMP=/opt/anaconda3 OMP_DYNAMIC=FALSE THREADS=4 bash scripts/run_member_c.sh
```

Linux/GCC 省略 LIBOMP。旧 OMP_PREFIX 保留为别名；推荐使用远端的 LIBOMP 参数。完整测试、调优、线程扩展、sanitizer 和提交包命令见提交说明。

## 与 B / D 的边界

B 已合入 main。C 的索引/尾部交叉验证覆盖三算法共用参考、非整除 N/d、tile=1/tile>N、奇数和过量线程。C 不需要等待 B/D 才能编译运行。

D 可直接使用当前 `bench.py`、`check.py --tune` 和线程扩展目标。继续在选定正式实验机器上冻结 tile、记录 affinity/频率并完成完整性能网格和报告。已有测量基础不等同于这些正式实验已全部完成。共享文件不再用旧 C 快照覆盖远端版本。
