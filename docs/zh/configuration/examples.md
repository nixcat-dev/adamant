# 配置示例

[English](../../configuration/examples.md)

用 `bin/basalt serve --config basalt_config.json` 启动。模型路径相对于该文件。设置说明：[配置](index.md)。

## 一块 GPU（32 GB）加 RAM

```json
{
  "version": 2,
  "models": [{ "model": "model.basalt", "gpu": "0", "max-context": 262144, "kv": "int8" }],
  "engine": { "resident-experts": true, "vram-reserve-mib": 700 },
  "server": { "host": "127.0.0.1", "port": 8080 }
}
```

## 一块 GPU 加一块较小的第二卡（对等专家层）

与第一个相同，把 `"gpu"` 写成 `"0,1"`，加 `"vision": 1`，图像编码器也放到第二张卡上跑。

## 显存较小的卡

与第一个相同，把 `"max-context"` 设为 `65536`、`"kv"` 设为 `"q4_0"`、`"vram-reserve-mib"` 设为 `1024`。

多个模型：[模型与路由](models.md)。条目里的 `"relaxed-drafting": 0.5` 用采样下损失精确性换取速度
（[精度](../precision/qwen38-flash-next.md#宽松起草)）。
