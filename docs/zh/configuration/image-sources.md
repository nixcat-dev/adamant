# 图像来源

[English](../../configuration/image-sources.md)

请求用 `data:` URL、`http(s)` URL、`file:` URL 或路径指定图像。服务器读取什么取决于请求是否来自本机。

| 来源 | 来自本机 | 来自其他地方 |
|---|---|---|
| `data:` URL | 可以 | 可以 |
| `http(s)` URL | 可以，任意主机 | 只有用 `--image-url-host` 命名的主机 |
| 路径或 `file:` URL | 可以，服务器用户可读的任意文件 | 只有 `--image-dir` 命名的目录下 |

每种来源的限制：每张图 32 MiB、64 百万像素，每次读取 60 秒，只限普通文件，每个请求 1 GiB 文件。

**本机**指回环对端、没有转发头、没有非本地的 `Host`，且 `Origin` 或 `Sec-Fetch-Site` 为 `none`（任何网页都算远程，
包括服务器自己的页面）。`--image-remote-only` 让所有请求都算远程。

- `--image-url-host HOST[:PORT]`：正好是那个主机。域名只能解析到公网地址。IP 地址按字面取，所以 `192.168.1.20:8000`
  对网络里的一台机器可用。
- `--image-dir DIR`：解析后的路径（经过符号链接和 `..` 之后）必须仍在 `DIR` 下。

**转发器：** 本机上把别人的请求转发的任何东西（默认配置的 nginx、`ssh -L`、`socat`、LiteLLM 之类）都会让它们看起来来自
本机。在那里设置 `--image-remote-only`。Caddy、Apache、Traefik 和 `tailscale serve` 会加上转发头。

**共享机器：** 本机请求以服务器用户的身份读文件。这类机器要设置 `--api-key`。

**容器：** 发布端口的客户端看起来是远程。用 `data:` URL，或者上面两个标志。
