# 雷达基础知识手册

![code size](https://img.shields.io/github/languages/code-size/ivaquero/book-radar.svg)
![repo size](https://img.shields.io/github/repo-size/ivaquero/book-radar.svg)

## 主要内容

- 雷达方程
- 波束形成
- 信号处理
- 弹道外推
- 雷达标校
- 组网技术

## 文档编译

本文档编写使用现代文本工具 [Typst](https://github.com/typst/typst)，其安装及使用方法可参考[知乎帖子](https://zhuanlan.zhihu.com/p/642509853)。

### 依赖包

- Typst
  - [ctheorems](https://github.com/sahasatvik/typst-theorems): v1.1.2
  - [phyisca](https://github.com/leedehai/typst-physics): v0.9.3
  - [fletcher](https://github.com/Jollywatt/typst-fletcher): v0.4.3
  - [hydra](https://github.com/tingerrr/hydra): v0.4.0
  - [gentle-clues](https://github.com/jomaway/typst-gentle-clues): v0.8.0
  - [indenta](https://github.com/flaribbit/indenta): v0.0.2

为保证正常编译，请参考 [typst-packages](https://github.com/typst/packages) 上的说明，在如下路径下克隆 `https://github.com/typst/packages` 仓库

- Linux：
  - `$XDG_DATA_HOME/typst`
  - `~/.local/share/typst`
- macOS：`~/Library/Application Support/typst`
- Windows：`%APPDATA%/typst`
