#import "lib/lib.typ": *
#show: chapter-style.with(
  title: "信号处理",
  info: info,
)

= 滤波器

== 传递函数

传递函数描述输入信号到输出信号的映射。对纯放大的简单情况，幅度 g 的变化称为传递函数的“增益”。即

$ y = g x $

对离散值信号，这意味着输出仅取决于瞬时输入

$ y(t) = g x(t) $

当系统的增益覆盖多个数量级时，幅度的相对变化通常以分贝（dB）为单位

$ G = 20 log_10 (frac(a_("out"), a_("int"))) "dB" $

== 滤波器类型

在大多数情况下，线性滤波器是线性时不变（LTI）滤波器。线性的必要条件是“输入加倍，输出加倍”。

- 线性
  - 叠加：系统对输入信号的响应是输入信号的响应之和。
  - 缩放：系统对输入信号的响应是输入信号的响应的缩放。
- 时不变：系统系数在特定时间内恒定。\

线性滤波器的叠加和缩放特性允许采用两种独立的方式来分析 LTI 滤波器：

- 输入视图：从给定的输入开始，并研究该输入对后续输出的影响
- 输出视图：分析输入的哪些部分对给定的输出值做出了贡献

== 有限脉冲响应滤波器

对$k$阶 FIR 滤波器，输出由最后$k + 1$个输入的加权和确定，权重为$w_k$

$ y(n) = ∑_(i = 0)^k w_i x(n - i) $ <fir>

这可以看作是一个移动窗口滤波器（moving window filter），它在输入数据点上从头到尾移动。这类滤波器通常被称为“有限脉冲响应”（FIR）滤波器。

@fir 有时也称为信号$x$与卷积核$𝒘$的卷积，写作

$ y(n) = w(k) * x(n) $

== 无限脉冲响应滤波器

虽然 FIR 滤波器的输出仅取决于输入信号（@fir），但滤波器的一般输出也可能取决于输出信号的$m$个最新值：

$ ∑_(j = 0)^m a_j * y(n - j) = ∑_(i = 0)^k b_i * x(n - i) $ <iir>

其中，$a_0 = 1$。系数$a_i$和$b_j$确定唯一的滤波器，这类滤波器通常被称为“无限脉冲响应”（IIR）滤波器。也可以写作

$ y(n) = b(k) * x(n) - a(k) * y(n) $

虽然为 IIR 滤波器找到所需的系数$(a, b)$比 FIR 滤波器更困难，但 IIR 滤波器的优点是计算效率更高，并且可以用更少的系数实现更好的频率选择。

```python
xx = np.zeros(20)
xx[5] = 1
tt = np.arange(20)
data = {}
data["before"] = xx
data["after_fir"] = signal.lfilter(np.ones(5) / 5, 1, xx)
data["after_iir"] = signal.lfilter([1], [1, -0.5], xx)

_, ax = plt.subplots(figsize=(6, 4))
ax.plot(tt, data["before"], "o", label="input")
ax.plot(tt, data["after_fir"], "x-", label="FIR-filtered")
ax.plot(tt, data["after_iir"], ".:", label="IIR-filtered")
```

#figure(
  image("images/filter-fir-iir.png", width: 40%),
  caption: "FIR 和 IIR 滤波器",
)

== 形态滤波器

FIR 滤波器和 IIR 滤波器都是线性滤波器，因为滤波器系数仅线性地进入传递函数。虽然这种线性滤波器擅长消除 Gaussian 分布的噪声，但它无法完成其他任务，如用于消除离群值。数据中的此类峰值可能是由以下原因引起的：由于传感器故障或实验装置中的连接松动。

对此类任务，所谓的“形态滤波器”可能很有用。这些形态滤波器不使用输入/输出的加权平均值，而是使用数据特征，如数据窗口内的元素的最值、中值、众数等。简单说，为了消除离群值，中值滤波器提供比线性滤波器更好的噪声抑制，其在平滑信号的同时，能保留信号的边缘，不会像线性滤波器那样去试图拟合离群值。

```python
x1 = np.zeros(20)
x1[10:] = 1
x1[[5, 15]] = 3
x_med = signal.medfilt(x1, 3)
b = np.ones(3) / 3
x_filt = signal.lfilter(b, 1, x1)

_, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(x1, "o", linestyle="dotted", label="rawdata")
ax1.plot(x_filt[1:], label="average")
ax1.plot(x_med, label="median")
```

#figure(
  image("images/filter-morph.png", width: 40%),
  caption: "形态滤波器",
)

= 移动平均平滑

平滑可以在数据中发现关键特征，并忽略噪声。

== 移动平均滤波器

长度为$N$的移动平均滤波器的输出是输入信号的最近$N$个值的平均值。即

$
  y[k]=frac(x[k] + x[k-1] + x[k-2]+ … + x[k-M+1], M)
$

请注意，长度为$N$的任何对称滤波器都会引入$(N-1)\/2$个采样延迟，最终结果中需要减去。

```python
coef_24h = np.ones(24) / 24
zi = signal.lfilter_zi(coef_24h, 1) * 0
avg24hTempC, _ = signal.lfilter(coef_24h, 1, tempC["tempC"].flatten(), zi=zi)
days = (np.arange(31 * 24) + 1) / 24
fDelay = (coef_24h.size - 1) / 2

_, ax_ma = plt.subplots(figsize=(8, 4))
ax_ma.plot(days, tempC["tempC"], label="Hourly Temp")
ax_ma.plot(days, avg24hTempC, label="24 Hour Average (delayed)")
ax_ma.plot(days - fDelay / 24, avg24hTempC, label="24 Hour Average")
```

#figure(
  image("images/filter-ma.png", width: 50%),
  caption: "移动平均滤波器",
)

可以通过最值来表征数据的变化。

```python
highIdx, _ = signal.find_peaks(tempC["tempC"].flatten(), distance=16)
envHigh = interpolate.UnivariateSpline(
    days[highIdx], tempC["tempC"][highIdx].flatten(), s=0
)
lowIdx, _ = signal.find_peaks(-tempC["tempC"].flatten(), distance=16)
envLow = interpolate.UnivariateSpline(
    days[lowIdx], tempC["tempC"][lowIdx].flatten(), s=0
)
envMean = (envHigh(days) + envLow(days)) / 2

_, ax_p = plt.subplots()
ax_p.plot(days, tempC["tempC"], label="Hourly Temp")
ax_p.plot(days, envHigh(days), label="High")
ax_p.plot(days, envMean, label="Mean")
ax_p.plot(days, envLow(days), label="Low")
```

#figure(
  image("images/filter-peaks.png", width: 50%),
  caption: "包络图",
)

== 二项式滤波器

一种非均匀采样的滤波器是遵循$[1\/2, 1\/2]^n$的二项式滤波器。对于大于$n$的值，这种滤波器逼近正态曲线；对于小于$n$的值，这种滤波器适合滤除高频噪声。要找到二项式滤波器的系数，需要对$[1\/2, 1\/2]$进行自身卷积，然后用$[1\/2, 1\/2]$与输出以迭代方式进行指定次数的卷积。

```python
h_b = np.array([1 / 2, 1 / 2])
binomCoef = signal.convolve(h_b, h_b)
for n in range(4): # 5次迭代
    binomCoef = signal.convolve(binomCoef, h_b)
zi_b = signal.lfilter_zi(binomCoef, 1) * 0
binomMA, _ = signal.lfilter(binomCoef, 1, tempC["tempC"].flatten(), zi=zi_b)

fDelay_b = (len(binomCoef) - 1) / 2
_, ax_b = plt.subplots(figsize=(8, 4))
ax_b.plot(days, tempC["tempC"], label="Hourly Temp")
ax_b.plot(
    days - fDelay_b / 24,
    binomMA,
    label="Binomial Weighted Average",
)
ax_b.legend()
```

#figure(
  image("images/filter-binom.png", width: 50%),
  caption: "二项式滤波器",
)

== 指数滤波器

另一种非均匀采样滤波器是指数移动平均滤波器。这种类型的滤波器易于构造，并且不需要窗大小。可以通过介于0和1之间的$α$参数来调整指数加权移动平均滤波器。$α$值越高，平滑度越低。

```python
alpha = 0.45
zi_ex = signal.lfilter_zi(np.array([alpha]), np.array([1, alpha - 1])) * 0
exponMA, _ = signal.lfilter(
    np.array([alpha]),
    np.array([1, alpha - 1]),
    tempC["tempC"].flatten(),
    zi=zi_ex,
)

_, ax_ex = plt.subplots()
ax_ex.plot(days, tempC["tempC"], label="Hourly Temp")
ax_ex.plot(days - fDelay / 24, binomMA, label="Binomial Weighted Average")
ax_ex.plot(days - 1 / 24, exponMA, label="Expon Weighted Average")
```

#figure(
  image("images/filter-expon.png", width: 50%),
  caption: "指数滤波器",
)

= 定期采样数据

== Savitzky-Golay 滤波器

移动平均滤波器快速且简单，但其输出不是很准确。如，它系统地低估了信号峰值附近的真实信号值。这个问题可以通过具有平滑和微分能力的 FIR 滤波器，Savitzky-Golay 滤波器@savitzkySmoothingDifferentiationData1964 来消除。该滤波器尝试以最小二乘方式对指定数量的采样进行指定阶数的多项式拟合。

Savitzky-Golay 滤波器将$q$阶多项式拟合到周围$2m + 1$个数据点，并使用其中心的值作为该点的输出。为了计算输入信号的一阶导数，采用拟合多项式中心的倾角；对二阶导数，采用曲率等。Savitzky-Golay 滤波器的优点之一是只需确定滤波器系数$w$一次，因为其不依赖于输入信号。
\
#block(
  height: 13em,
  columns()[
    Savitzky-Golay 滤波器的主要步骤为
    + 选定$x$值
    + 选择一个关于$x$的对称窗口
    + 计算与这些数据拟合的最佳多项式
    + 取点
      - 对平滑，取拟合曲线的中心点；
      - 对一阶导数，取该位置的切线；
      - 对二阶导数、取曲率；
    Savitzky–Golay 滤波器需要以下参数
    + 数据窗口的大小（奇数）
    + 多项式拟合的阶数（严格小于窗口大小）
    + 导数阶数（0为平滑，1为导数）
    + 采样率（Hz）
  ],
)

- Savitzky-Golay 滤波器的优点：计算高阶导数非常方便；平滑和求导可以同时进行。
- Savitzky-Golay 滤波器的缺点：没有清晰的频率响应。即，增益仅随着频率的增加而逐渐减小。
  - 若只有 < 200 Hz 的频率分量应通滤波波器，则最好采用其他滤波技术，如 Butterworth 滤波器。
  - 若真实信号的理想信号特征已知，Wiener 滤波器@wienerInterpolationExtrapolationSmoothing1942 虽然应用更复杂，但可能会产生更好的结果。

```python
cubicMA = signal.savgol_filter(tempC["tempC"].flatten(), 7, 3)
quarticMA = signal.savgol_filter(tempC["tempC"].flatten(), 7, 4)
quinticMA = signal.savgol_filter(tempC["tempC"].flatten(), 9, 5)

_, ax_sg = plt.subplots(figsize=(8, 4))
ax_sg.plot(days, tempC["tempC"], label="Hourly Temp")
ax_sg.plot(days, cubicMA, label="Cubic-Weighted MA")
ax_sg.plot(days, quarticMA, label="Quartic-Weighted MA")
ax_sg.plot(days, quinticMA, label="Quintic-Weighted MA")
```

#figure(
  image("images/filter-savgol.png", width: 50%),
  caption: "Savitzky-Golay 滤波器",
)

不难看出，数据长度一致时，阶数越小，曲线越平滑；数据长度越大，曲线越平滑。

== 微分

数据的微分是一项常见的任务。它可用于查找与给定位置信号相对应的速度和加速度、查找极值、确定曲线的切线以及许多其他应用。

=== 一阶差分

输入信号的微分由下式给出

$ y(n) = frac(Δ x, Δ t) = frac(x(n) - x(n - 1), Δ t) $

这给出了 FIR 滤波器的滤波器权重

$ 𝒘 = [1, -1] \/ Δ t $

=== 中心差分微分器

对离线分析，居中滤波器可能更合适

$ 𝒘 = [1, 0, -1] * frac(1, 2 * Δ t) $

=== 三次微分器

我们还可以通过在每个点之前和之后获取两个样本，并获取中心数据点的最佳三次拟合的斜率来区分曲线（这是 Savitzky-Golay 滤波器的特例）。这可以通过权重向量来实现

$
  𝒘 = [1, -8, 0, 8, -1] * frac(1, 12 * Δ t)
$

== 积分

=== 解析积分

用于运动记录的典型测量设备提供速度或加速度。如，惯性测量单元（inertial measurement units，IMU）通常提供物体的线性加速度。

为了分别从线加速度和速度获得线速度和位置，必须对这些数据进行积分：

$
  "vel"(t) &= "vel"(t_0) + ∫_(t_0)^t "acc"(t^′) dd(t^′)
$ <vel>

$
  𝒙(t) &= 𝒙(t_0) + ∫_(t_0)^t "vel"(t^(′′)) dd(t^(′′)) = \
  &= 𝒙(t_0) + "vel"(t_0) * (t - t_0) + ∫_(t_0)^t ∫_(t_0)^(t^(′′)) "acc"(t^′) dd(t^′, t^(′′))
$ <acc>

若传感器在$t_0$处的速度为$v_0$，则位置变化由下式给出

$
  Δ 𝒙(t) = 𝒙(t) - 𝒙(t_0) = 𝒗_0 * Δ t + ∫_(t_0)^t ∫_(t_0)^(t^(′′)) "acc"(t^′) dd(t^′, t^(′′))
$ <pos>

方程中的加速度。 @vel ∼ @pos 是相对空间的加速度。然而，在使用惯性传感器进行测量时，加速度是相对传感器获得的。

=== 数值积分

当处理离散数据时，积分只能近似确定。将$t_0$和$t$之间的时间分成$n$个宽度为$Δ t$的相等元素，结果是

$
  𝒙(t_n) = ∑_(i = 1)^n 𝒙_i
$

测量离散时间$t_i (i = 0, ..., n)$处的加速度，@vel2 和@acc2 必须用离散方程代替：

$
  "vel"(t_(i + 1)) ≈ "vel"(t_i) + "acc"(t_i) * Δ t
$ <vel2>

$
  𝒙(t_(i + 1)) ≈ 𝒙(t_i) + "vel"(t_i) * Δ t + frac("acc"(t_i), 2) * Δ t^2
$ <acc2>

从视觉上看，曲线的积分由曲线下的面积给出。虽然积分可以作为 IIR 滤波器执行，但使用函数提供的一阶和二阶近似更容易。

#figure(
  image("images/filter-integ.jpg", width: 40%),
  caption: "对位置数据进行数值积分以获得速度",
)

= 不定期采样数据

曲线平滑是一个重要的主题，根据要求可以使用不同的解决方案。平滑选项取决于例如数据的采样方式。若它们以相等的间隔采样，则可以使用 FIR 或 IIR 滤波器。但若它们是以不同的时间延迟记录的，如无线传感器，并且偶尔丢失了样本，则需要其他方法。

== Loess 和 Lowess 平滑

平滑不规则间隔的一维数据的两种常见方法是 loess 滤波器（LOcal regrESSion）和 lowess 滤波器（LOcally WEighted Scatterplot Smoothing）。这两种是相关的非参数回归方法，将多个回归模型与所谓的“基于 k 最近邻的元模型”相结合。loess 是 lowess 的泛化。

简而言之，指定要包含的相邻数据的百分比。对这些数据，应用加权线性回归。用于 lowess 和 loess 的传统权函数是三立方（tri-cube）权函数。

$
  w(x) =(1 -|x|^3)^3 𝑰_[|x| < 1 ]
$

其中，$𝑰[...]$是指示函数，指示函数参数为 True 的范围：在此范围内函数等于 1，否则为 lowess 和 loess 的不同之处在于它们用于回归的模型：lowess 使用线性多项式，而 loess 使用二次多项式。

== 样条

样条（spline）不仅可用于插值，还可用于数据平滑和微分。如今，样条函数被定义为变量$x$中次数$< k$的分段多项式函数。 $k$称为样条的次数（degree），$k + 1$称为样条的阶数（order）。

== B-Spline

构建平滑、分段多项式 2-D 和 3-D 轨迹的一种特别简单而强大的选项是所谓的 B 样条。术语 B 样条代表“基础样条”，因为给定次数的任何样条函数都可以表示为该次数的 B 样条的线性组合。

对给定的轨迹，样条结（spline-knots）将轨迹的分段多项式部分分开。若结是等距的，则该样条称为基数 B 样条（cardinal B-spline），并且 B 样条的定义变得非常简单：具有$p$次 B 样条（$p ∈ N_0$）、卷积算子$*$和指示函数半开单位区间$[0, 1)$的$b^0 = 𝑰_([0, 1))$，相应的基数 B 样条由下式给出

$
  𝒃^𝒑 := underbrace(𝑰_([0,1)) * ⋯ * 𝑰_([0,1)), p+1-"times")
$

#tip[
  B 样条具有所谓的最小支持：线性 B 样条仅对两个相邻的结起作用，二次 B 样条对三个结起作用，等等。
]

#figure(
  image("images/filter-spline-b.png", width: 40%),
  caption: "B 样条",
) <bspline>

一个等级$p$的 B 样条$C(u)，u ∈ [τ_p, τ_(n−p−1)]$，，具有结向量$τ$和控制点（也称 DeBoor 点）$P_i (i = 0, ..., n − p − 2)$由下式给出

$
  C(u) = ∑_(i = 0)^(n - p - 2) P_i B_(i, p)(u)
$

其中，$B_(i, p)$ 是移至$i$点的@bspline 中的$b^p$。由于按照惯例，@bspline 中的第一个样条$b^0$被称为“1 阶 B 样条”，因此“阶 n”B 样条包含$n − 1$次多项式。

== 核密度估计

在许多给出了离散数据样本的应用中，数据需要平滑的“密度分布”。如，对一维数据，事件频率通常用直方图表示，而对二维数据，事件频率通常用散点图表示。为了从这些数据中获得平滑的概率密度函数，可以使用核密度估计（KDE）技术。

对于一维数据的 KDE，每个样本都乘以 Gaussian 分布的 PDF 函数，然后对所得曲线进行求和。

#figure(
  image("images/filter-kde.png", width: 40%),
  caption: "直方图 vs 核密度估计",
)

= 图像滤波

== 图像的色彩

最简单的图像类型是灰度图像：每个像素都有一个与其亮度相对应的灰度级。对于许多图像来说，8 位灰度级分辨率就足够了，可提供$2^8 = 256$个灰度级。请注意，只有当所使用的传感设备以更精细的水平测量差异时，更高的图像深度才有意义。

#tip[
  Python 浮点数通常使用 64 位。由于这需要的内存是无符号 8 位的 8 倍，因此只有在真正需要时才应将数据转换为浮点型。
]

彩色图像只不过是三个灰度图像的堆叠：一个代表“红色”通道，一个代表“绿色”通道，一个代表“蓝色”通道。若三种颜色按此顺序堆叠，则该图像称为 RGB 图像。

== 图像的透明度

有时，彩色图像包含附加的第四层，称为“alpha 层”：该层指定该图像的透明度量。该技术称为“alpha 混合”或“alpha 合成”。

== 二维滤波

类似于 1D 中的 FIR 的滤波也可以对 2D 信号（如图像）执行。我们现在有一个 2D 输入和一个权重矩阵，而不是一维输入和权重向量。影响滤波器输出的像素周围的区域有时称为“结构元素”（structural element，SE）。它不一定是正方形，也可以是矩形、圆形或椭圆形。输出仍然是通过对输入的加权求和得到：

$ y(n, m) = ∑_(i = - k)^k ∑_(j = -l)^l w_(i j) * x(n + i, m + j) $

移动窗口解释仍然成立，除了：
- 窗口和数据在两个维度上扩展
- 内核索引的序列没有反转\

所以这更多地对应于 2D 互相关而不是 2D 卷积。

== 二维形态滤波器

线性滤波器是可交换的。换句话说，这些滤波器的应用顺序并不重要。然而，对数据的形态操作是非线性操作，最终结果取决于顺序。

#bibliography("lib/dsp.bib", title: "参考文献")
