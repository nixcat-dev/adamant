# 精度

[English](../../precision/index.md)

**Adamant：** 本移植的执行路径及其验证状态见 [Ada 的执行与验证](ada.md)，其中也列出 Ampere（sm_86）上未完成的工作。Adamant 自己的测量结果在
[README](../../../README.zh-CN.md)。下面的页面描述的是 Blackwell 上的上游 Basalt；其中的测量环境与数字
不是 Adamant 的结果。

每个模型一页，说明它以什么精度计算、每种舍入要付出多少代价。方法是共用的。

- [Qwen3.8-Flash-Next](qwen38-flash-next.md)：各阶段、prefill、实测成本、起草与有损标志。
- [质量如何测量](quality.md)：门槛、BF16 参考基准、贪心一致性与复现命令。
