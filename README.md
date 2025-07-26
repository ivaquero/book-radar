# 雷达基础知识手册

![code size](https://img.shields.io/github/languages/code-size/ivaquero/book-radar.svg)
![repo size](https://img.shields.io/github/repo-size/ivaquero/book-radar.svg)

## 主要内容

- 雷达方程
- 波束形成
- 信号处理
- 弹道外推
- 数据融合

> 使用 `radar.typ` 生成 PDF 文件。

## 文档编译

本文档编写使用现代文本工具 [Typst](https://github.com/typst/typst)，其安装及使用方法可参考[知乎帖子](https://zhuanlan.zhihu.com/p/642509853)。

### 克隆官方仓库

为保证正常编译，请参考 [typst-packages](https://github.com/typst/packages) 上的说明，在如下路径下克隆 `typst-packages` 仓库

- Linux：
  - `$XDG_DATA_HOME`
  - `~/.local/share`
- macOS：`~/Library/Application Support`
- Windows：`%APPDATA%`

```bash
cd [above-path]
git clone --depth 1 --branch main https://github.com/typst/packages typst
```

### 使用模版

```typst
#import "@preview/qooklet:0.5.1": *
```
