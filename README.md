# 雷达基础知识手册

![code size](https://img.shields.io/github/languages/code-size/ivaquero/book-radar.svg)
![repo size](https://img.shields.io/github/repo-size/ivaquero/book-radar.svg)

## 主要内容

- 雷达方程
- 波束形成
- 信号处理
- 弹道外推
- 数据融合

> 生成的 PDF 在 articles 文件夹。

## 文档编译

本文档编写使用现代文本工具 [Typst](https://github.com/typst/typst)，其安装及使用方法可参考[知乎帖子](https://zhuanlan.zhihu.com/p/642509853)。

### 依赖包

- Typst
  - [ctheorems](https://github.com/sahasatvik/typst-theorems)
  - [phyisca](https://github.com/leedehai/typst-physics)
  - [fletcher](https://github.com/Jollywatt/typst-fletcher)
  - [hydra](https://github.com/tingerrr/hydra)
  - [gentle-clues](https://github.com/jomaway/typst-gentle-clues)
  - [indenta](https://github.com/flaribbit/indenta)

为保证正常编译，请参考 [typst-packages](https://github.com/typst/packages) 上的说明，在如下路径下克隆 `https://github.com/typst/packages` 仓库

- Linux：
  - `$XDG_DATA_HOME/typst`
  - `~/.local/share/typst`
- macOS：`~/Library/Application Support/typst`
- Windows：`%APPDATA%/typst`
