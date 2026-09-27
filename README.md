# Latent Atlas

AI 概念的可交互图谱：用动画和画面，把 Transformer、推理引擎、Agent 工作流拆开来看。

## 现在有什么

- **Transformer / Forward pass**：GPT-2 small 一次完整的前向计算，从分词到 LM head 采样，可以拖动时间线、切换注意力头、调温度。
- **Transformer / Attention**：一个注意力头的全部矩阵乘法，按 toy 规模（d_model 8、d_head 4）用真实的算术逐格演示：X·W_Q/K/V → Q·Kᵀ → ÷√d → mask → softmax → A·V → concat·W_O → 加回残差。鼠标悬停在任意结果格子上，可以看到它由哪一行、哪一列算出来。

在总览页点击带 ↗ 的部件（目前是 attn 板），会放大进入对应的细节页。

## 开发

```bash
npm install
npm run dev              # http://localhost:5173
npm run build            # 类型检查 + 生产构建到 dist/
npm run build:artifact   # 单文件构建到 dist-artifact/index.html，用于发布为 claude.ai Artifact
```

## 结构

```
src/
  main.ts                 侧栏导航、hash 路由、放大/缩小的转场
  styles.css              设计 token（颜色、字体）和页面框架样式
  core/
    stage.ts              DPR 自适应的 canvas 和 rAF 循环
    player.ts             阶段时间线：播放、拖动、速度、键盘快捷键
    frame.ts              展品的页面框架：标题、规格、说明、控制栏
    draw.ts               共用的绘图原语：token chip、玻璃板、标签、数学符号
    theme.ts              从 CSS token 读取 canvas 调色板，跟随深浅色切换
  exhibits/
    registry.ts           门类和展品列表；有 route 的展品才会上线
    transformer/
      model.ts            toy 模型：GPT-2 token id、注意力的真实计算、采样分布
      overview.ts         前向总览
      attention.ts        注意力展开页
```

新增展品：在 `exhibits/` 下写一个 `mount(root, nav) => destroy` 函数，然后在 `registry.ts` 里给对应条目加上 `route` 和 `mount`。
