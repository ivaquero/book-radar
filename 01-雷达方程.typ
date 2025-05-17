#import "lib/lib.typ": *
#show: chapter-style.with(
  title: "雷达方程",
  info: info,
)

= 简介

雷达（Radar）是无线电探测和测距（radio detection and ranging）的缩写。在大多数情况下，雷达系统使用调制波形和定向天线向特定空间发射电磁能，以搜索目标。搜索范围内的目标会向雷达方向反射部分入射能量（雷达回波）。雷达接收器对这些回波进行处理，以提取目标信息，如距离、速度、角位置和其他目标识别特征。

= 波段

无线电频谱是电磁频谱中频率从 3 Hz 到 3,000 GHz 的部分。这一频率范围内的电磁波被称为无线电波，广泛应用于现代科技。为了防止不同用户之间的干扰，无线电波的产生和传输受到各国法律的严格管制，并由一个国际机构--国际电信联盟（ITU）进行协调。

历史上，雷达最初是作为军事工具开发的。正因为如此，最常见的雷达系统分类是二战期间和二战后军方最初使用的字母或波段名称，这种字母或波段名称也被 IEEE（电气与电子工程师协会）标准采用。近年来，北大西洋公约组织（NATO）采用了一种新的波段名称，其字母更简洁。

== 波段划分

#let data = csv("data/radar-freq.csv")
#figure(
  tableq(data, 11),
  caption: "频率与波段",
  supplement: "表",
  kind: table,
)

== 波段应用

雷达可分为陆基、机载、空载或舰载雷达系统。根据雷达的具体特性，如频段、天线类型和使用的波形，雷达还可分为许多类别。使用调制或其他连续波形的雷达系统被归类为连续波（Continuous Wave，CW）雷达，而使用有时间限制的脉冲波形的雷达系统被归类为脉冲雷达（Pulsed Radars）。

根据不同波段电磁波的特性，雷达被用于不同的场景。

#let data = csv("data/radar-waveband.csv")
#figure(
  tableq(data, 6),
  caption: "不同波段的应用",
  supplement: "表",
  kind: table,
)

= 雷达方程

== 基本式

$
  P_r &= frac(P_t G^2 λ^2 σ F, (4π)^3 R^4 L)\
  &= underbrace((P_t G_t) / (4π R^2), ctext("发射能量密度"))
  ⋅ underbrace(σ / L, ctext("有效反射面积"))
  ⋅ underbrace(F / (4π R^2), ctext("距离衰减"))
  ⋅ underbrace((G_r λ^2) / (4π), ctext("有效接收面积"))
$

其中，$P_t$和$P_r$分别为（雷达）峰值发射功率和峰值接收功率，$G_t$和$G_r$分别为（天线）发射增益和接收增益，$R$为（目标）探测距离，$σ$为雷达反射截面积（radar cross section，RCS），$L$为（系统）能量损失，$F$为传播因子，$λ$为信号波长。其中

$
  G_r = frac(ctext("定向功率密度"), ctext("同向功率密度")) = frac(A_("sphere"), A_("ant")) = frac(4π R^2, A_("ant")) ≈ frac(4π R^2, θ_("azi") θ_("ele")) = frac(4π R^2, (R λ) / b (R λ) / h) = frac(4π A, λ^2)
$

#let data = csv("data/radar-target.csv")
#figure(
  tableq(data, 5),
  caption: "目标特性",
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

#theorem(title: "Carson 法则")[
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
