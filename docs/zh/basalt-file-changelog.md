# `.basalt` 文件格式变更日志

[English](../basalt-file-changelog.md)

版本号是文件头部 magic 之后的那个 `u32`。引擎只读取一个版本；
[模型文件](model-file.md)页面描述的是当前版本。

## 版本 2

当前版本。引擎拒绝版本 1 文件，并指明
[tools/migrations/basalt_v1_to_v2.py](../../tools/migrations/README.md)，它能把版本 1 文件直接写成版本 2 文件，无需下载、无需重新量化。

- **包含：** `meta`、`tokenizer/*`、`dense`、`dense.index`、`experts`、`experts.layout`、`gguf`、`mtp/*`、`hc8`、
  `vision/*`。
- **新增 `hc8`，现为必需：** decode 读取的 hyper-connection 矩阵的 int8 副本，也就是引擎过去每次启动都要自己计算的位。
  启动时直接读取它们，省去读取 BF16 矩阵再量化的步骤（约半秒）。
- **移除 `profile`：** 专家 profile 决定哪些专家初始驻留 VRAM。它是可以独立于权重改进的放置策略，
  因此移到了引擎里（`data/profiles`），更好的 profile 不再需要新文件。
- **精简 `meta`：** 不再有 `sampling`（服务器读取 GGUF 元数据的 `general.sampling.*` 键）、不再有 `profile` 或
  `hc8` 记录，也没有转换器版本号——每次替换某个 section 时这些溯源信息都会变化。

## 版本 1

- **包含：** `meta`（带 `sampling`、`profile` 和 `hc8` 记录以及转换器的 git revision）、
  `tokenizer/*`、`dense`、`dense.index`、`experts`、`experts.layout`、`gguf`、`mtp/*`、`profile`、可选的 `hc8`、
  `vision/*`。
- 第一个单文件格式：引擎、服务器和 vision sidecar 加载的所有内容都在其中。没有 `hc8` 时，
  引擎每次启动都要量化 hyper-connection 矩阵。
