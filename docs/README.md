# GitHub Pages Configuration

## 部署说明

本仓库已配置GitHub Pages自动部署，将docs目录中的内容发布为静态网站。

### 文件结构

```text
book-radar/
├── docs/           # GitHub Pages 源文件
│   ├── beamf.html  # 波束形成可视化页面
│   ├── beamf.css   # 样式文件
│   ├── beamf.js    # JavaScript交互逻辑
│   └── formulae.js # 数学公式渲染
├── .github/
│   └── workflows/
│       └── pages.yml # GitHub Actions 部署配置
└── ...
```

### 功能特性

- **波束形成可视化**：交互式雷达波束形成演示
- **实时计算**：滑动条控制参数，实时更新计算结果
- **图形化展示**：极坐标图、相位图、阵列图等
- **响应式设计**：适配不同屏幕尺寸

### 访问地址

部署完成后，可通过以下地址访问：
`https://ivaquero.github.io/book-radar/`

### 自动部署

每次推送到main分支时，GitHub Actions会自动部署更新。
