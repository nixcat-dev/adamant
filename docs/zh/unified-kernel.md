# 统一 kernel

[English](../unified-kernel.md)

统一 kernel 工作中有一块留在了 engine 里；它引出的整层 kernel 与 GDN mid chain 都已退役。

## 在运行的部分

| 部件 | 来源 | 作用 |
|---|---|---|
| commit 收尾器 | `verify_kernels.cu` | GDN commit 在自己的 stream 上从一个计数器领取它的项目；下一个窗口以一个领走剩余部分的收尾器开始。配上设备规划的尾段与完整的 GDN mixer 时，commit stream 只承担第一个 GDN 层：更后的层在下一个窗口内部领取，各自先于自己的 GDN kernel，并在窗口等待 CPU 层时提前完成。同样的这些 launch 会在合并（combine）等待某个层（CPU 池，或该层没有 CPU 条目时对等卡的行）期间，把下一层的投影权重预读进 L2。 |

- 成对环的槽位数总是偶数，因为两组 tile 交替取 tile：奇数环会让某个槽位拿到来自不同组的连续 tile，某一组可能在数据到达之前就通过它的等待。parity 测试对每种类型三元组检查环大小。
- 算术与归约顺序与其取代的 kernel 相同：输出与状态逐位一致，无数值变化。
- 收尾器的交接等待设有上限：越过上限的等待会置起粘滞的 abort 字（`wait_abort.hpp`，映射的主机内存）并直接放行而不是 trap；此后每个等待一旦看到该字也直接放行，主机则将该窗口判为失败并不再运行后续窗口。

## 测试

- `dec_commit_parity` 与 `dec_commit_finisher`（`--finisher`）：在 commit 之前、之中、之后运行收尾器，附 microbench。
- `dec_commit_lazy`（`--lazy`）：窗口内 commit 对照 commit kernel 与 CPU 参考，以及它的预读，附 microbench。

## 已退役的部分

这个分支最初把整个层融进一个 kernel：attention 链、专家那一半（plan、gate/up、SwiGLU、down、共享专家、层级等待、combine）以及 QSA attention 那一半。输出与分开跑的 kernel 逐位相同，但 decode 比分开跑更慢，所以只有 GDN 中段与收尾器留了下来。

同一会话，IQ3_S pack，`--peer-exclusive`，prose 样例，speculation 7，GPU 400 W（agent 开发 A/B）：

| 构建 | 窗口（ms） | tokens/s |
|---|---|---|
| 分开跑的 kernel | 6.675 | 324.6 |
| 整层 kernel | 7.541 | 288.2 |

T = 3 时，按每个 GDN 层计，attention 那一半与分开跑的 kernel 持平；专家那一半输了约 11 us（plan + gate/up +21 us，该半其余部分 -11 us）。QSA 那一半每层输约 35 us（投影 +13，输出投影 +11，hc2 + router +6）。折算到每个窗口，各约 +0.4 ms。

它输在哪里：

- 专家那一半受限于跨 CTA 依赖的延迟（gate/up 第二轮，然后 h 行，然后 down），不是 DRAM：T = 3 时约 58 us，而字节下界约 30 us。
- QSA 那一半把分开跑的 kernel 会与旁路 kernel 重叠执行的投影串行化，还新增了 grid 屏障。
- 每个 SM 放不下两个 CTA（99 KB arena，512 线程各 128 个寄存器）。
- 层 kernel 的 SASS 约 1.5 MB；gate/up 的 warp 卡在取指上（20% 的 PC 采样）。按窗口宽度分别编译把它降到 1.13 MB 与每窗口约 1%，而 gate/up 阶段的结束时间没有变化。
- 一行以及五到七行的窗口毫无收益，而融合层需要关掉 tag 协议，这在五到七行的窗口上花掉约 1%。
- 独立 harness 预测的收益在 engine 里没有出现（比率 0.1 到 0.3）：它让每个 CTA 启动时专家就已确定，而在 engine 里，它们在 attention 输出之后约 7 us 才到达。

持久 QSA attention 与 kernel 内的阶段打点（`BASALT_PROFILE`）随之一起移除。

GDN mid chain 随后也退役了：它是一次持久 launch，把 qkv + z 成对、递推与输出投影放在该层两次超连接读取之间。它要求整张卡的 CTA 全部驻留，并要求一个编译单元通过 `FZ_DEV_ONLY` 守卫复用其他 kernel 的设备代码来编译。GDN 层的中段又恢复为依次执行 `dense_tc_pair`、`dec_gdn_rec` 与 `dense_tc`。

## 经验教训

- 在真实的 engine 二进制上检查寄存器溢出（`cuobjdump --dump-resource-usage`）。对这些编译单元用 `nvcc --split-compile` 会让层 kernel 的溢出多出约四倍，tokens/s 损失约 9%。这里绝不要用它。
- 装不下动态共享内存的 kernel 会静默回退；计时之前先确认融合的快路径真的在跑。
- 计时要持有 GPU 锁、没有构建在跑、热循环里没有逐 launch 的 memset；打点数字只在同一会话内比较，ABAB。
- 用窗口宽度分布把每层收益换算成窗口收益（每层微秒约 0.73 x 36）。
- harness 必须先复现 engine 的起始状态（迟到的 id、层级等待、共享专家尾段），其差值才有预测力。
- 用 CUPTI 连续 PC 采样剖析停顿原因，并通过 `-lineinfo` 构建上的 `nvdisasm -gi` 映射回源码。
