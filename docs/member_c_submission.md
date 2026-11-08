# C 提交清单与 A/B/D 集成说明

更新于 2026-11-08。C 分支基于上游 `855179728f95b755974f7a1ee5ecc6c462bb8771`。B、`bench.py`、统一 ISA/SIMD/no-vec 和线程扩展入口由上游提供。

## 1. C 的贡献文件

| 文件 | C 负责的增量 |
| --- | --- |
| `streaming.cpp` | online softmax、K packing、八分数 SIMD、尾部处理、独立 worker scratch、OpenMP 和参数检查 |
| `tests/streaming_test.cpp` | 201/601 组 API 配置，输入保护、脏缓冲区、复用、SIMD 边界、溢出、线程一致性 |
| `scripts/run_member_c.sh` | 统一构建 + 三算法正确性 + 上游 bench.py 小型比较 + 内存报告 |
| `scripts/memory_report.py` | 对当前 A/B/C 核对 12 形状共 36 项内存容量 |
| `benchmarks/streaming_original.cpp` | 冻结的旧 C，仅用于消融，不参与普通构建 |
| `benchmarks/compare.cpp`、`scripts/benchmark_streaming.py` | C 的旧版/优化版/消融比较；独立固定 flags，不能替代当前团队 benchmark |
| `scripts/profile_streaming.py` | 临时源码插桩诊断；不修改生产源码 |
| `scripts/package_member_c.py` | 白名单打包、相对当前上游的 C 补丁、哈希和证据清单 |
| `docs/member_c.md`、`docs/member_c_submission.md` | 设计、内存、当前操作方式及贡献边界 |
| `docs/streaming_optimization.md` | 2026-10-17 历史优化记录，明确旧环境/旧比较范围 |
| `docs/integration_2026-11-08.md` | 此次 A/B/C 集成与实测验证记录 |

公共文件只认领以下修改，不认领整份文件：

| 文件 | C 的修改 | 保留的上游内容 |
| --- | --- | --- |
| `attention.h` | C 实现状态、scratch 与 packing 注释 | 公共 API 和配置 |
| `main.cpp` | `--memory` 查询入口 | 新版 `--build-info`、三算法分发、计时与输入输出 |
| `check.py` | C 的极大有限值溢出拒绝案例 | FP64 reference、tuning、线程一致性、正确性框架 |
| `Makefile` | C API 目标、可选 sanitizer、LIBOMP 兼容、带空格 PYTHON 路径、线程测试失败传播 | 上游 ISA/SIMD、no-vec、bench/thread-scaling/vec-report 基础 |
| `README.md`、`.gitignore` | C 导航和集成状态、忽略 C 产物 | 上游使用说明和已有 AI 辅助声明 |

`attention.cpp`、`blocked.cpp`、`simd.h`、`bench.py`、`requirements.txt`、`.github/workflows/validation.yml` 沿用远端，不算 C 新增。README 原有 AI 辅助说明继续保留。

## 2. 一条命令与新机器运行

以下命令在仓库或解压后根目录执行。新机器需要 C++17、Make、Python 3.12/3.13；先安装固定依赖：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
bash scripts/run_member_c.sh
```

当前本地环境已安装，直接执行最后一行。默认串行不依赖 OpenMP。脚本输出：

- `results/member_c_quick.csv` / `.log`：三算法正确性与边界检查。
- `results/member_c_bench.csv` / `.machine.json` / `.log`：调用新版 bench.py 的小型 vec/no-vec 比较。
- `results/member_c_memory.csv`：36 项内存查询。
- `results/member_c_environment.txt` / `member_c_build.log`：实际二进制 flags、依赖与编译命令。

重跑会覆盖同名文件。历史 C 优化结果位于独立的 `results/optimization/`，此次集成证据位于 `results/integration_2026-11-08/`。

当前 Mac（已有 Anaconda libomp）：

```sh
OPENMP=1 LIBOMP=/opt/anaconda3 OMP_DYNAMIC=FALSE THREADS=4 bash scripts/run_member_c.sh
```

Linux/GCC：

```sh
OPENMP=1 OMP_DYNAMIC=FALSE THREADS=4 bash scripts/run_member_c.sh
```

其他 Mac 的 LIBOMP 指向实际安装，例如 `/opt/homebrew/opt/libomp`。旧 OMP_PREFIX 是兼容别名。切换 flags 会触发重新构建；运行前用 `./attention --build-info` 核对。

## 3. 完整正确性、调优和正式实验入口

```sh
mkdir -p results
make PYTHON=.venv/bin/python
./attention --list
.venv/bin/python check.py --algorithm all --full --tiles --csv results/full_correctness.csv
```

算法列表必须包含 baseline、blocked、streaming，`all` 只遍历实际编译的实现。完整三算法性能测试使用上游 bench.py，不再使用 `check.py --bench`。

```sh
.venv/bin/python check.py --algorithm all --tune --tiles --tuning-output results/tuned-tiles.json --csv results/tuning.csv
make PYTHON=.venv/bin/python attention attention-novec
.venv/bin/python bench.py --full --threads 1 --rounds 3 --tile-config results/tuned-tiles.json --novec-binary ./attention-novec --csv results/ladder.csv
```

在 Mac 上进行线程正确性和扩展实验：

```sh
make PYTHON=.venv/bin/python OPENMP=1 LIBOMP=/opt/anaconda3 attention attention-novec test-streaming-api
.venv/bin/python check.py --algorithm all --full --tiles --threads 4 --csv results/full_t4.csv
make PYTHON=.venv/bin/python OPENMP=1 LIBOMP=/opt/anaconda3 test-threads
OMP_PLACES=cores OMP_PROC_BIND=close .venv/bin/python bench.py --threads 1,2,4 --shapes 512x64,4096x64 --rounds 3 --tile-config results/tuned-tiles.json --csv results/scaling.csv
```

Linux 命令省略 LIBOMP。实际扩展上限应根据选定机器物理核数确定；不能由示例中的 4 线程代替完整扩展实验。

```sh
./attention 4096 64 --memory --algorithm streaming --br 32 --bc 32
.venv/bin/python scripts/memory_report.py --csv results/memory.csv
make PYTHON=.venv/bin/python sanitize
```

Sanitizer 默认启用 ASan 和 UBSan。Mac 上使用独立的 LLVM 22.1.8 工具链通过了完整参考检查和 C 的 601 组 API 测试。Linux ARM64 容器通过了边界及 API 补测。详见[集成记录](integration_2026-11-08.md)。若仅运行 UBSan，可指定 `SANITIZERS=undefined`。

```sh
make PYTHON=.venv/bin/python OPENMP=1 LIBOMP=/opt/anaconda3 SANITIZERS=undefined sanitize
```

## 4. 后续 B/D 协作

- B 已完成的实现与 C 使用同一 attention.h；scratch 查询单位为 float 数量，统一 runner 换算字节。A/B scratch 是当前完整 N×N 矩阵；C 需计入 K panel，公式见设计文档。
- C 的每行 recurrence 固定由一个 worker 处理。D 复用当前 OpenMP 入口，不再重复套并行循环；动态 team、绑核、物理核与频率策略应记录到正式实验。
- 所有算法先用同一 FP64 reference 验证，再计时；packing、初始化与线程开销包括在 call 内。统一 flags 不代表 A/B/C 使用相同算术实现，不能把所有家族差异归因于单一优化。
- C 的历史五版本消融只研究 C 的优化。团队的三算法比较、no-vec、调优和线程扩展统一走新版 bench.py/check.py。
- 本机已完成全形状 benchmark 和两组形状的线程扩展。目标服务器的 tile、性能和独立的数据竞争检测按团队实验计划验证。

## 5. 提交包和备份

```sh
python3 scripts/package_member_c.py
```

生成 `dist/member_c_submission.zip`：包含 A/B/C 可运行快照、C 相对 `8551797` 的补丁、归属与 SHA-256 清单、历史和当前证据。不包含环境、编译产物、Git 元数据或未列入清单的文件。`--code-only` 可省略证据。

新补丁必须在匹配基线上先执行 `git apply --check`，不能在已含改动的目录重复应用。后续上游继续变化时，应重新对照公共文件，不要用 ZIP 覆盖队友代码。

集成前的本地备份位于 `dist/pre_integration_2026-11-08/`。
