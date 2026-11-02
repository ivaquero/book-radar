# 雷达基础知识手册

![code size](https://img.shields.io/github/languages/code-size/ivaquero/book-radar.svg)
![repo size](https://img.shields.io/github/repo-size/ivaquero/book-radar.svg)

## 主要内容

- 雷达方程
- 波束形成
- 信号处理（滤波器 / 特征查找 / 频谱分析）
- 空间配准
- 时间配准
- MATLAB 雷达相关工具箱 API

> 使用 `book-radar.typ` 生成 PDF 文件。

## 文档编译

本文档编写使用现代文本工具 [Typst](https://github.com/typst/typst)，其安装及使用方法可参考[知乎帖子](https://zhuanlan.zhihu.com/p/642509853)。

### 克隆官方仓库

为保证正常编译，请参考 [typst-packages](https://github.com/typst/packages) 上的说明，在如下路径下克隆 `typst-packages` 仓库

- Linux：
  - `$XDG_DATA_HOME`
  - `~/.local/share`
- macOS：`~/Library/Application Support`
- Windows：`%APPDATA%`

```shell
cd [above-path]
git clone --depth 1 --branch main https://github.com/typst/packages typst
```

### 使用模版

```typst
#import "@preview/qooklet:0.7.2": *
```

## 本地编译

```shell
typst compile book-radar.typ --font-path "C:/Windows/Fonts"
```

## 图片再生成

`images/` 由 `python/` 下的三个 [marimo](https://marimo.io) notebook 生成（`dsp_filtering.py` → filter-*，`dsp_finding.py` → find-*，`dsp_analysis.py` → spectral-*）。默认 `SAVE_FIGS = False` 只显示不落盘；置为 `True` 后运行 notebook 即覆盖 `images/`：

```shell
pip install -r python/requirements.txt
marimo run python/dsp_filtering.py
```
