#import "lib/sci-book.typ": *
#show: doc => conf(
  title: "雷达方程",
  author: ("SZZX"),
  footer-cap: "SZZX",
  header-cap: "雷达基础知识手册",
  outline-on: false,
  doc,
)

= 简介

雷达（Radar）是无线电探测和测距（radio detection and ranging）的缩写。在大多数情况下，雷达系统使用调制波形和定向天线向特定空间发射电磁能，以搜索目标。搜索范围内的目标会向雷达方向反射部分入射能量（雷达回波）。雷达接收器对这些回波进行处理，以提取目标信息，如距离、速度、角位置和其他目标识别特征。

= 波段

无线电频谱是电磁频谱中频率从 3 赫兹到 3,000 千兆赫 的部分。这一频率范围内的电磁波被称为无线电波，广泛应用于现代科技。为了防止不同用户之间的干扰，无线电波的产生和传输受到各国法律的严格管制，并由一个国际机构--国际电信联盟（ITU）进行协调。

历史上，雷达最初是作为军事工具开发的。正因为如此，最常见的雷达系统分类是二战期间和二战后军方最初使用的字母或波段名称，这种字母或波段名称也被 IEEE（电气与电子工程师协会）标准采用。近年来，北大西洋公约组织（NATO）采用了一种新的波段名称，其字母更简洁。

== 波段划分

#let l3 = $3×10^5$
#let m3 = 3 * calc.pow(10, 8) / (3 * calc.pow(10, 6))
#let m30 = 3 * calc.pow(10, 8) / (30 * calc.pow(10, 6))
#let m250 = 3 * calc.pow(10, 8) / (250 * calc.pow(10, 6))
#let m300 = 3 * calc.pow(10, 8) / (300 * calc.pow(10, 6))
#let m500 = 3 * calc.pow(10, 8) / (500 * calc.pow(10, 6))
#let g1 = 3 * calc.pow(10, 8) / (1000 * calc.pow(10, 6))
#let g2 = 3 * calc.pow(10, 8) / (2000 * calc.pow(10, 6))
#let g3 = 3 * calc.pow(10, 8) / (3000 * calc.pow(10, 6))
#let g4 = 3 * calc.pow(10, 8) / (4 * calc.pow(10, 9)) * calc.pow(10, 3)
#let g6 = 3 * calc.pow(10, 8) / (6 * calc.pow(10, 9)) * calc.pow(10, 3)
#let g8 = 3 * calc.pow(10, 8) / (8 * calc.pow(10, 9)) * calc.pow(10, 3)
#let g10 = 3 * calc.pow(10, 8) / (10 * calc.pow(10, 9)) * calc.pow(10, 3)
#let g12 = 3 * calc.pow(10, 8) / (12 * calc.pow(10, 9)) * calc.pow(10, 3)
#let g18 = calc.round(
  3 * calc.pow(10, 8) / (18 * calc.pow(10, 9)) * calc.pow(10, 3),
  digits: 1,
)
#let g20 = 3 * calc.pow(10, 8) / (20 * calc.pow(10, 9)) * calc.pow(10, 3)
#let g27 = calc.round(
  3 * calc.pow(10, 8) / (27 * calc.pow(10, 9)) * calc.pow(10, 3),
  digits: 1,
)
#let g30 = 3 * calc.pow(10, 8) / (30 * calc.pow(10, 9)) * calc.pow(10, 3)
#let g40 = 3 * calc.pow(10, 8) / (40 * calc.pow(10, 9)) * calc.pow(10, 3)
#let g60 = 3 * calc.pow(10, 8) / (60 * calc.pow(10, 9)) * calc.pow(10, 3)
#let g75 = 3 * calc.pow(10, 8) / (75 * calc.pow(10, 9)) * calc.pow(10, 3)
#let g100 = 3 * calc.pow(10, 8) / (100 * calc.pow(10, 9)) * calc.pow(10, 3)

#figure(
  table(
    columns: 8,
    align: center,
    inset: 4pt,
    stroke: frame(rgb("000")),
    table.header(
      [*频率*],
      [*波长*],
      [*IEEE*],
      [*NATO*],
      [*频率*],
      [*波长*],
      [*IEEE*],
      [*NATO*],
    ),
    [3Hz],
    [#l3 km],
    table.cell(rowspan: 2)[HF],
    table.cell(rowspan: 3)[A],
    [8GHz],
    [#eval(g8)mm],
    table.cell(rowspan: 3)[X],
    [I],
    [3MHz],
    [#eval(m3)m],
    [10GHz],
    [#eval(g10)mm],
    table.cell(rowspan: 4)[J],
    [30MHz],
    [#eval(m30)m],
    table.cell(rowspan: 2)[VHF],
    [12GHz],
    [#eval(g12)mm],
    [250MHz],
    [#eval(m250)m],
    table.cell(rowspan: 2)[B],
    [18GHz],
    [#eval(g18)mm],
    [Kᵤ],
    [300MHz],
    [#eval(m300)m],
    table.cell(rowspan: 2)[UHF],
    [20GHz],
    [#eval(g20)mm],
    table.cell(rowspan: 2)[K],
    [500MHz],
    [#eval(m500)m],
    [C],
    [27GHz],
    [#eval(g27)mm],
    table.cell(rowspan: 3)[K],
    [1GHz],
    [#eval(g1)m],
    [L],
    [D],
    [30GHz],
    [#eval(g30)mm],
    table.cell(rowspan: 2)[Kₐ],
    [2GHz],
    [#eval(g2)m],
    table.cell(rowspan: 2)[S],
    [E],
    [40GHz],
    [#eval(g40)mm],
    [3GHz],
    [#eval(g3)m],
    [F],
    [60GHz],
    [#eval(g60)mm],
    table.cell(rowspan: 2)[V],
    [L],
    [4GHz],
    [#eval(g4)mm],
    table.cell(rowspan: 2)[C],
    [G],
    [75GHz],
    [#eval(g75)mm],
    table.cell(rowspan: 2)[M],
    [6GHz],
    [#eval(g6)mm],
    [H],
    [100GHz],
    [#eval(g100)mm],
    [W],
  ),
  caption: [频率与波段],
  supplement: "表",
  kind: table,
)

#pagebreak()

== 波段应用

雷达可分为陆基、机载、空载或舰载雷达系统。根据雷达的具体特性，如频段、天线类型和使用的波形，雷达还可分为许多类别。使用调制或其他连续波形的雷达系统被归类为连续波（Continuous Wave，CW）雷达，而使用有时间限制的脉冲波形的雷达系统被归类为脉冲雷达（Pulsed Radars）。

根据不同波段电磁波的特性，雷达被用于不同的场景。

#let csv1 = csv("data/waveband.csv")
#figure(
  ktable(csv1, 5),
  caption: [不同波段的应用],
  supplement: "表",
  kind: table,
)

= 雷达方程

== 基本式

$
  P_r &= frac(P_t G^2 λ^2 σ F, (4π)^3 R^4 L)\
  &= underbrace((P_t G_t)/(4π R^2), "发射能量密度")
  ⋅ underbrace(σ/L, "有效反射面积")
  ⋅ underbrace(F/(4π R^2), "距离衰减")
  ⋅ underbrace((G_r λ^2)/(4π), "有效接收面积")
$

其中，$P_t$和$P_r$分别为（雷达）峰值发射功率和峰值接收功率，$G_t$和$G_r$分别为（天线）发射增益和接收增益，$R$为（目标）探测距离，$σ$为雷达反射截面积（radar cross section，RCS），$L$为（系统）能量损失，$F$为传播因子，$λ$为信号波长。其中

$
  G_r = frac("定向功率密度", "同向功率密度") = frac(A_("sphere"), A_("ant")) = frac(4π R^2, A_("ant")) ≈ frac(4π R^2, θ_("azi") θ_("ele")) = frac(4π R^2, (R λ)/b (R λ)/h) = frac(4π A, λ^2)
$

#pagebreak()

#let csv1 = csv("data/target.csv")
#figure(
  ktable(csv1, 4),
  caption: [目标特性],
  supplement: "表",
  kind: table,
)

== 噪声

由接收噪声功率（received noise power，RNP）

$ N = k_B T_s B_N $

其中，$k_B$为 Boltzmann 常数，$T_s$为系统噪声温度，$B_N$为信号的噪声带宽。可得信噪比为

$
  "SNR" = frac("Signal Power", "Noise Power") = frac(P_t G λ^2 σ F, (4π)^3 R^4 L k_B T_s B_N)
$

不难看出，与探测距离

- 成反比的有：信噪比、噪声带宽、噪声温度、系统能量损失
- 成正比的有：峰值发射功率、增益、信号波长

== 脉冲

$
  "range" = frac(c Δ t, 2)\
  "radial velocity" = frac(Δ x, τ)
$

其中，$c$为光速，$τ$脉冲间隔。

== 频率

=== Doppler 效应

$
  "mix" = sin (a t) ⋅ sin (b t)\
  "mix" = 1 / 2 [cos(a - b) t - cos(a + b) t]
$

其中，$b-a$称为拍频（beat frequency）又称差频，指两个频率相近但不同的声波的干涉，所得到的干涉信号的频率是原先两个声波的频率之差的绝对值。雷达中，拍频通常由 Doppler 效应产生，是传输频率（transmission frequency，TF）和接收频率（reception frequency，RF）之间的差值。

=== 调频

设计雷达时，在探测距离达到要求的情况下，还应考虑距离分辨率。

对于静止的物体，差频为$0$，信号沿水平方向移动，会严重影响测量精度，这个时候需要对频率进行调频（modulation）。

=== 线性调频

线性调制是将传输频率增加到另一个值，创建锯齿状调制模式。当接收信号有延迟时，每时每刻会得到不同的频率。由产生的差频，可以计算出量测的位

#rule("Carson 法则")[
  几乎所有（≈98%）的调频信号的功率处于带宽。
]

== 矩形脉冲

= 辅助知识

== 物理

#definition[
  Boltzmann 常数是一个比例因子，它将气体中粒子的平均相对热能与气体的热力学温度联系起来。其在数值上等于气体常数$R$和 Avogadro 常数$N_A$之比

  $ k_B = frac(R, N_A) $

  新的 Boltzmann 常数已被 ISO 设定为

  $ k_B = 1.380649 × 10^(-23) J / K $
]
