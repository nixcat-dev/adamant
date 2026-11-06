# 示例配置

[English](../../configuration/examples.md)

设置的说明见[配置](index.md)。模型路径相对于配置文件。用 `bin/basalt serve --config basalt_config.json` 启动服务器。

## 一块 GPU（32 GB）加 RAM

显卡装不下的专家留在页锁定 RAM 中。

```json
{
  "server": {
    "model": "model.basalt",
    "gpu": "0",
    "host": "127.0.0.1",
    "port": 8080
  },
  "engine": {
    "max-context": 262144,
    "kv": "int8",
    "resident-experts": true,
    "vram-reserve-mib": 700
  }
}
```

## 一块 GPU 加一块更小的第二卡作为对等专家层

列出两块卡会让第二块成为对等层。`"vision": 1` 让图像编码器也跑在那块卡上。

```json
{
  "server": {
    "model": "model.basalt",
    "gpu": "0,1",
    "host": "127.0.0.1",
    "port": 8080,
    "vision": 1
  },
  "engine": {
    "max-context": 262144,
    "kv": "int8",
    "resident-experts": true,
    "vram-reserve-mib": 700
  }
}
```

## 一块低显存的卡

更小的 KV 格式和更短的上下文为专家留出更多 VRAM；其余留在 RAM 中。

```json
{
  "server": {
    "model": "model.basalt",
    "gpu": "0",
    "host": "127.0.0.1",
    "port": 8080
  },
  "engine": {
    "max-context": 65536,
    "kv": "q4_0",
    "resident-experts": true,
    "vram-reserve-mib": 1024
  }
}
```

以上任何一个都可以选配：在 `engine` 里加 `"relaxed-drafting": 0.5`，用采样下的精确性换取速度（见
[投机解码](index.md#投机解码-mtp)）。
