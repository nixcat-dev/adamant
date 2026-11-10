# 模型与路由

[English](../../configuration/models.md)

多个模型共用一个服务器。请求的 `model` 字段选其中一个，同一时间只加载一个。换模型会结束一个引擎进程并启动下一个。

```json
{
  "version": 2,
  "engine": { "conversation-cache-disk-mib": 32768, "idle-sleep": 300, "vram-reserve-mib": 500 },
  "models": [
    { "alias": "Flash-Next", "model": "flash-next.basalt", "gpu": "auto", "kv": "int8", "max-context": 262144, "parallel": 4,
      "slot-full-context": true, "vision": "dynamic" },
    { "alias": "Flash-Next-Q8", "model": "flash-next-q8.basalt", "gpu": "0,1", "kv": "fp16", "max-context": 65536 }
  ],
  "server": { "host": "127.0.0.1", "port": 8080 }
}
```

- `models`：每个条目有一个 `model`（相对于该文件）和一个可选的 `alias`，即 API 报告的名字。
- `engine`：对所有接受它的模型生效。条目自己的值优先。
- `server`：整个服务器共享：`host`、`port`、`allowed-host`、`cors-origin`、`image-url-host`、`image-dir`、
  `image-remote-only`、`lazy`、`open`、`debug`、`request-lines`。
- 条目接受引擎标志以及这些服务器选项：`gpu`、`vision`、`parallel`、`slot-full-context`、`idle-sleep`、
  `idle-unload`、`min-free-vram-mib`、`fit-max-tokens`、`grammar`、推理相关设置和采样默认值。
- `version` 是 `2`。旧形状的文件（没有 `models`）会被拒绝。
- 命令行对每个模型都优先。

## 启动时的检查

每个条目在加载任何东西之前都会按引擎自己的规则检查：

| | 模型条目或命令行 | 顶层 `engine` |
|---|---|---|
| 模型没有的标志 | 停止启动 | 对该模型略过；启动行会说明 |
| 模型拒绝的取值（`max-context` 超过它自己的上限） | 停止启动 | 停止启动 |
| 未知名字 | 停止启动 | 停止启动 |

`kv` 取模型自己的取值（`fp16`、`int8`、`q4_0`、`k8v4`）；其他取值会停止启动。两个相同的 alias，或两个条目共用同一个缓存
目录，也会停止启动。

## 请求与轮次

请求的 `model` 是 alias 或文件的文件名，在任何 API 上都一样。其他名字都是 404。`/v1/models` 显示 `loaded`、
`loading`、`unloaded` 或 `failed`。

客户端会等待换模型，不会因此得到 503。等待期间流式响应每 10 秒发一条 keep-alive 注释。

- 没有任何请求在等待时，对已加载模型的请求立刻运行。一旦有另一个模型的请求在等待，后来的请求就排在它后面。
- 队首决定下一个模型；它的换模型在运行中的请求结束后开始。该模型在那一刻一次取走为它排队的全部请求，作为一个批次。
- 加载失败时用 503 命名该模型回应它的批次，队列继续。一次加载会尝试两次。只有一个模型时，启动加载失败会结束服务器。

## 换模型保留什么

| | 换模型之后 |
|---|---|
| 磁盘上的会话 | 保留在各模型自己的目录 |
| RAM 和槽位中的会话 | `conversation-cache-disk-mib` 开启时在退出时写盘；否则丢失 |
| 睡眠中的引擎 | 下一个发给它的请求把它唤醒 |
| `POST /v1/vram` | 属于整台机器；对每个能调整自身专家层的模型生效 |
| 指标 | 每个模型各自（`GET /metrics?model=NAME`） |

`POST /v1/load` 带 `{"model": NAME}` 会抢在请求之前换模型。`/health` 和 `/metrics` 显示换模型的进度。控制台打印
`[server] swap: Flash-Next -> Flash-Next-Q8, 2 waiting`。
