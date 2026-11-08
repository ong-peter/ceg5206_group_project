# 2026-11-08 A/B/C 集成记录

## 集成状态

C 分支基于上游 `855179728f95b755974f7a1ee5ecc6c462bb8771`，即 2026-11-07 的 `add blocked threading, and scaling sweep`。该基线已包含 B。

集成前公共文件的改动保留在带日期说明的 Git stash 中；全部本地源码另备份到 `dist/pre_integration_2026-11-08/local_changes.zip`。已核对 `streaming.cpp` 与备份逐字节相同，本次没有重写算法或改变容差。

## 合并内容

- 保留上游 `attention.cpp`、`blocked.cpp`、`simd.h`、`bench.py`，不认领为 C 的新增代码。
- 保留上游 `main.cpp --build-info`，合入 C 的 `--memory`，二者均可用。
- 保留上游调优、线程一致性与正确性框架，只合入 C 原有的溢出输入拒绝检查。
- 保留上游 native ISA、SIMD/no-vec、线程和 benchmark 目标，添加 C API 与 sanitizer 测试目标。
- 保留 LIBOMP 作为 macOS OpenMP 参数，并兼容旧 OMP_PREFIX；补充运行库搜索路径。Makefile 的 PYTHON 调用均加引号，以支持本项目带空格的路径。
- 修正 `test-threads` 的失败传播：不再让 `tail` 的成功状态掩盖 Python 测试失败。注入失败检查器后，目标按预期非零退出。
- 一键脚本不再传 `check.py --bench`；正确性用 check.py，计时用新版 bench.py，显式检查三算法内存报告。
- 提交清单、README、设计文档和打包基线已更新。10 月的优化报告保留为历史记录，不能与当前编译条件的数据混合计算 speedup。

## 实测验证

当前证据目录为 `results/integration_2026-11-08/`；汇总与源文件指纹在 `validation_summary.json`。不是沿用旧报告来声明本次通过。

| 项目 | 本次结果 | 证据 |
| --- | --- | --- |
| A/B/C 单线程，全部 12 主形状+边界 | A 21、B 111、C 111 项通过 | `correctness_serial.csv` / `.log` |
| OpenMP 4 线程，同一完整集合；baseline 维持 1 线程 | A 21、B 111、C 111 项通过；B/C 与 1 线程逐位一致 | `correctness_openmp4.csv` / `.log` |
| 关闭向量化的三算法完整集合 | A 21、B 111、C 111 项通过 | `correctness_novec.csv` / `.log` |
| C 直接 API | 单线程 201、多线程 601 组通过 | `build_serial.log`、`build_openmp.log` |
| B/C 特殊线程边界 | 40 组通过：2/3/4/14/64 线程，含线程多于 query block、N=1 与非整除维度 | `thread_edges.log` |
| 多线程构建 UBSan | 参考测试及 C 601 组 API 检查通过 | `ubsan.log` |
| 线程测试失败传播 | 注入失败检查器，make 如预期失败 | `thread_failure_guard.log` |
| 三算法内存查询 | 单线程和 4 线程各 36 项通过 | `memory_serial.csv`、`memory_openmp4.csv` |
| 上游 tile tuning | B/C 在两个形状、九组 tile 共 36 项候选通过验证与计时；选中配置后参考测试通过 | `tuning.csv`、`tuned-tiles.json`、`tuning.log` |
| 新版 bench.py | 两个代表形状，vec/no-vec、1/4 线程、两轮交错比较；先校验后计时 | `bench.csv`、`bench.machine.json`、`bench.log` |
| 更新后一键入口 | 串行、OpenMP 4 线程均运行成功 | `reproduction_serial.log`、`reproduction_openmp4.log` |

C 在完整参考检查中的最大绝对误差为 `1.1178691438741908e-5`；逐元素容差仍为 `1e-4 + 1e-3*abs(Oref)`。没有放宽精度要求。新 SIMD 设置下 B 允许不同于逐项归约的舍入，但也满足相同规则。

当前调优运行选择 B/C 均为 64×64，选择依据来自上游的两个形状归一化时间几何平均。这是本机这次测量的结果，不更改公共默认 32×32，也不自动冻结为其他机器的最优值。基准读取了该 JSON，验证了配置加载链路。

### 后续验证

使用独立的 LLVM 22.1.8 工具链在 Apple Silicon Mac 原生运行 ASan 和 UBSan。故意越界的探针报告了堆缓冲区越界。项目随后通过全部 12 个主形状及边界用例，共 A 21、B 111、C 111 项参考检查。B/C 的 4 线程输出与各自单线程逐位一致。C 的 601 组 API 配置也通过了检查。项目测试期间未出现 ASan 或 UBSan 报告。Linux ARM64 容器另通过边界和 601 组 API 检查。原始日志保存在本地 `results/asan_retest_20261108_210821/`。

本机扩展 benchmark 覆盖 12 个形状、96 个配置，并在 `512×64` 和 `4096×64` 上测量请求 1、2、4、8、14 线程时的表现。该轮调优选出 B 的 32×16 和 C 的 64×16。原始数据保存在本地 `results/extended_test_20261108_203128/`。

## 运行方式

```sh
bash scripts/run_member_c.sh
OPENMP=1 LIBOMP=/opt/anaconda3 OMP_DYNAMIC=FALSE THREADS=4 bash scripts/run_member_c.sh
```

第一条使用串行；第二条适用于当前安装了 Anaconda libomp 的 Mac。Linux/GCC 省略 LIBOMP。完整移植命令和贡献归属见 [提交说明](member_c_submission.md)。

## 结论与限制

C 与 A/B 共用接口、构建、参考验证、内存查询、调优配置和统一 benchmark 入口。

性能数据来自本地 Mac。目标服务器上的性能实验和独立的数据竞争检测仍待验证。

A/B/C 的 SIMD/packing 算术并非完全相同，家族间时间差不是纯粹的 online-softmax 消融。后续共享 microkernel 或严谨归因实验仍需保留各自正确性、内存和公平计时约定。
