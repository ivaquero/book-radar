<<<<<<< HEAD
#import "@local/scibook:0.1.0": *
#show: doc => conf(
  title: "信号处理2",
  author: "SZZX",
  footer-cap: "SZZX",
  header-cap: "雷达基础知识手册",
  outline-on: false,
  doc,
)

= 滤波器特性

== 脉冲和阶跃响应

- 脉冲响应：信号从0到1再到0的过程
- 阶跃响应：信号从0到1的过程

#figure(
  image("images/filter-response.png", width: 40%),
  caption: "脉冲和阶跃响应",
  supplement: [图],
)

=== 频率响应

LTI 传递函数的一个重要特性是，具有给定频率的正弦输入总是会产生具有相同频率的正弦输出，仅修改相位和/或幅度。传递增益的幅度和相位可以方便地表示为复数：幅度对应于复数的长度，相位对应于角度。这可用于表征滤波器响应。

#figure(
  image("images/filter-response-freq.png", width: 40%),
  caption: "频率响应特征",
  supplement: [图],
) <freqz>

@freqz 的 x 轴包含归一化频率，其中 1 对应于 Nyquist 频率。根据 Nyquist-Shannon 采样定理，信号中的信息可忠实再现的频率上限为

$ f_("Nyquist") = f_("sample") / 2 $

#figure(
  image("images/filter-response-nyq.png", width: 40%),
  caption: "采样再现",
  supplement: [图],
)

== 因果滤波器的伪影

因果滤波器的初始瞬态是由于滤波器窗口比已有的可用输入数据长引起的。按惯例，相应的缺失输入值设置为0，并且在数据分析中最好忽略瞬态。另一个影响是输出数据的延迟，这直接关系到数据的实时分析。

#definition[
  伪影（artifacts）：滤波器初始化值显示的“初始瞬态”，其输出相对输入总是存在时间延迟。
]

== 非因果滤波器的伪影

对离线分析，使用以居中分析窗口（centered analysis window）通常更方便。这消除了滤波后的输出信号$y$相对输入信号$x$的时间延迟问题。在 DSP 语言中，这称为非因果滤波器，因为未来点包含在$y_i$的评估中，这在数据的实时分析中是不可能的。

= 常用预处理技术

== 去趋势

去趋势是一种常见的预处理技术，用于消除数据中的趋势。趋势是数据中的长期变化，通常由数据中的周期性或线性变化引起。去趋势的目的是消除这种长期变化，以便更好地分析数据的短期变化。

#sgrid(
  figure(
    image("images/find-trend.png", width: 90%),
    caption: "趋势数据",
  ),
  figure(
    image("images/find-trend2.png", width: 90%),
    caption: "去趋势数据",
  ),
  columns: (200pt,) * 2,
  gutter: 2pt,
  caption: "",
)

```python
dt_ecgl = signal.detrend(
    ecgl["ecgl"], axis=-1, type="linear", bp=0, overwrite_data=False
)
dt_ecgnl = signal.detrend(
    ecgnl["ecgnl"], axis=-1, type="linear", bp=0, overwrite_data=False
)
```

可以看到，这些趋势已有效地去除，可以看到与原始信号相比，信号的基线已不再偏移，它们现在可用于进一步处理。

== 去干扰

=== Butterworth 陷波滤波器

Butterworth 滤波器旨在在通带中具有尽可能平坦的频率响应。因此，它也被称为“最大平坦幅度”滤波器。它可以用作低通、高通或带通滤波器。

这里的陷波宽度定义为 59 至 61 Hz 的频率区间。滤波器至少去除该范围内频率分量的一半功率。

```python
Fs = 1000
nom, denom = signal.iirfilter(
    N=5, Wn=[59, 61], btype="bandstop", analog=False, ftype="butter", fs=Fs
)
freq, h = signal.freqz(nom, denom, fs=Fs)
```

#figure(
  image("images/find-butterworth.png", width: 40%),
  caption: "Butterworth 滤波器",
  supplement: [图],
)

#warning[
  警告：请小心低滤波器频率的高阶（n ≥ 4）Butterworth 滤波器，其中，$[b,a]$语法可能会因舍入误差而导致数值问题。在这种情况下，应使用 IIR 滤波器推荐的“SOS”（second-order sections）语法或“ZPK”（zero-pole-gain）语法。
]

- 根据应用的不同，其他频率响应可能更可取。如，Chebyshev 滤波器提供比 Butterworth 滤波器更锐利的频率响应，而 Bessel 滤波器的优点是在时域中不会出现过冲（overshoot）。
- 为了将数据拟合到参数模型，使用原始数据几乎总是比使用预平滑数据更好，因为平滑已经丢弃了可用信息。

=== Filtfilt 滤波器

`Filtfilt` 是常见的非因果滤波器，其通过对数据运行滤波器两次来避免这种延迟：一次从前到后，一次从后到前。

```python
x_volt = openLoop["openLoopVoltage"]
t_volt = np.arange(len(x_volt)) / Fs
buttLoop = signal.filtfilt(nom, denom, x_volt.flatten())
```

#figure(
  image("images/find-filtfilter.png", width: 40%),
  caption: "Filtfilt 滤波器",
  supplement: [图],
)

#tip[
  要小心，滤波器的多次应用会改变滤波器特性。
]

== 重采样

如果构造一个均匀加权的移动平均滤波器，它将去除相对于滤波器持续时间而言具有周期性的任何分量。以 1000 Hz 采样时，在 60 Hz 的完整周期内，大约有$1000\/60=16.667$个采样。尝试“向上舍入”并使用一个 17 点滤波器。这将在$1000\/17=58.82$Hz的基频下为我们提供最大滤波效果。

#figure(
  image("images/find-savgol.png", width: 40%),
  caption: "Savitzky-Golay 滤波器",
  supplement: [图],
)

然而，虽然电压明显经过平滑处理，但它仍然包含小的 60 Hz 波纹。如果对信号进行重采样，以便通过移动平均滤波器捕获 60 Hz 信号的完整周期，就可以显著减弱该波纹。如果以$17*60=1020$Hz对信号进行重采样，可以使用17点移动平均滤波器来去除 60 Hz 的电线噪声。

#figure(
  image("images/find-resample.png", width: 40%),
  caption: "重采样",
  supplement: [图],
)

== 去峰值

=== Hampel 滤波器

中值滤波器会去除尖峰，但同时也会去除原始信号的大量数据点。Hampel 滤波器的工作原理类似于中值滤波器，但它仅替换与局部中位数值相差几倍标准差的值。

```python
def hampel(x, k, n_sigma=3):
    """
    x: pandas series of values from which to remove outliers
    k: size of window/2
    n_sigma: number of standard deviations to use; 3 is standard
    """
    arraySize = len(x)
    idx = np.arange(arraySize)
    output_x = x.copy()
    output_idx = np.zeros_like(x)

    for i in range(arraySize):
        mask1 = np.where(idx >= (idx[i] - k), True, False)
        mask2 = np.where(idx <= (idx[i] + k), True, False)
        kernel = np.logical_and(mask1, mask2)
        median = np.median(x[kernel])
        std = 1.4826 * np.median(np.abs(x[kernel] - median))
        if np.abs(x[i] - median) > n_sigma * std:
            output_idx[i] = 1
            output_x[i] = median

    return output_x, output_idx.astype(bool)
```

#sgrid(
  figure(
    image("images/filter-med.png", width: 90%),
    caption: "中值滤波器",
  ),
  figure(
    image("images/filter-hampel.png", width: 90%),
    caption: "Hampel 滤波器",
  ),
  columns: (200pt,) * 2,
  gutter: 2pt,
  caption: "",
)

= 事件和特征查找

通常需要在数据流中查找特定位置。如，人们可能想知道信号何时超过给定阈值。或者，在分析运动信号时，人们可能想找出运动开始的点和结束的点。若数据是时间序列，这些位置通常称为事件（event）。

== 查找简单的特征

真实的测量数据总是包含噪声（并且通常比您想要的多得多）。为了从分析中消除噪声影响，从连续数据到二进制真/假表示通常会很有帮助。

例如，可以通过以下分析步骤来找到位置记录中的移动周期

+ 找到一个阈值
+ 对于每个数据点，计算信号是否高于阈值（真/假）。
+ 查找特征

== 插值

当找到曲线穿过给定阈值的点时，我们可能需要比数据中更高的精度。为此，我们可以通过插值找到记录样本之间的数据点。

=== 线性插值

最简单的插值形式是线性插值，其中样本之间的点是通过相邻样本之间的线性连接获得的。虽然这是一种计算速度快的方法，但它有两个缺点：

- 这不是很准确
- 所得曲线的一阶导数在样本位置处不连续

=== 三次样条插值

线性插值的缺点可以通过三次样条插值来克服。样条表明多项式系数满足两个条件：

- 它们在样本位置是连续的
- 每个多项式末尾的导数在一定阶数下是连续的

#figure(
  image("images/find-interpol.png", width: 40%),
  caption: "插值",
  supplement: [图],
)

#tip[
  插值样条与 B 样条明显不同：前者总是经过给定的数据点，而后者则被数据吸引。
]

= 测度

== 互相关

在信号处理中，互相关（cross-correlation）是对两个序列的相似性的度量，作为一个序列相对于另一个序列的位移的函数。这也称为滑动点积或滑动内积。它通常用于在长信号中搜索较短的已知特征。互相关本质上类似于两个函数的卷积。

为了计算互相关，较短的特征通常用零填充以获得与较长信号相同的长度$n$。按照这种约定，$x$和$y$的互相关函数可以通过以下方式获得

$
  R_(x y)(m) = cases(
    ∑_(i = 0)^(n - m - 1) x_(m + i) y_i^* quad & 0 ≤ m < n - 1,
    R_(y x)^* (-m) quad &1 - n < m < 0
  )
$ <cross_corr>

其中，$x$和$y$是长度为$n (n > 1)$的复向量，$y^*$是$y$的复共轭。式中互相关$R_(x y)$的定义不是固定的，并且可以根据$x$和$y$的序列以及输入值为复数时共轭的元素而变化。为了优化速度，通常不直接使用@cross_corr 来实现互相关，而是通过快速 Fourier 变换（Fast Fourier Transform，FFT）。

== 比较信号

让我们考虑如何评估两个信号的相似性，我们将其称为信号和特征。为了找到相似性，我们需要某种“相似性函数”，当特征与信号匹配时，该函数具有最大值，并且随着信号和特征之间的差异增加而减小。

#figure(
  image("images/find-crosscorr.png", width: 40%),
  caption: "互相关",
  supplement: [图],
) <cross_corr2>

将信号的一部分与特征进行比较所需要做的就是将信号的该部分与特征相乘。若想找出特征需要移动多少才能匹配信号，我们可以计算不同相对移动的相似度，并选择相似度最大的移动。对于与给定信号/特征范围之外的元素的乘法（@cross_corr2），相应的丢失数据通常被$0$替换。

#figure(
  table(
    columns: (auto,) * 9,
    inset: 0.3em,
    align: left + horizon,
    stroke: frame2(rgb("000")),
    [0], [0], [*2*], [*1*], [*4*], [*3*], [0], [0], [],
    [1], [3], [2], [], [], [], [], [], [$arrow.r 2^* 2 = 4$],
    [], [1], [3], [2], [], [], [], [], [$arrow.r 2^* 3 + 1^* 2 = 8$],
    [], [], [1], [3], [2], [], [], [], [$arrow.r 2^* 1 + 1^* 3 + 4^* 2= 13$],
    [], [], [], [1], [3], [2], [], [], [$arrow.r 1^* 1 + 4^* 3 + 3^* 2= 19$],
    [], [], [], [], [1], [3], [2], [], [$arrow.r 4^* 1 + 3^* 3 = 13$],
    [], [], [], [], [], [1], [3], [2], [$arrow.r 3^* 1 = 3$],
  ),
  caption: "互相关",
  supplement: [表],
  kind: table,
)

总而言之，互相关提供了两条信息：

- 信号和特征的相似度（互相关的最大值）
- 信号和特征之间的相对移动（互相关的最大值位置）

=== 自相关

若比较的两个信号相同，则结果称为自相关（auto-correlation）。当然，自相关函数不用于查找事件。相反，它可以用于检测信号中的周期性（periodicity）。在考虑平均偏移之后，自相关还用于检测信号中的能量，因为谐波信号中的能量与幅度的平方成正比。

在自相关（即信号与其自身的互相关）中，始终会在零滞后处出现峰值，其大小就是信号能量。

#figure(
  image("images/find-autocorr.png", width: 40%),
  caption: "自相关",
  supplement: [图],
)

=== 归一化

为了评估信号的形状（无论其总体持续时间、幅度或偏移量如何），我们必须对信号进行归一化。因此，归一化必须考虑信号的三个方面：

- 偏移（offset）：为了消除恒定偏移的影响，我们可以减去信号的平均值，从而避免三角形伪影，或者减去信号的最小值，从而确保互相关的输出始终为正；
- 持续（duration）：为了确保两个信号具有相同的长度，我们可以将它们插值到固定数量的数据点；
- 幅度：标准化互相关信号幅度的最常见方法是使自相关函数的最大值 = 1。这可以通过以下方式实现

$
  "sig"_("norm") = frac("sig"_("raw"), sqrt(max("autocorr"("sig"_("raw")))))
$ <sig_norm>

通过减去最小值和用方程归一化的幅度来消除偏移的信号。 @sig_norm 的幅度介于$0$和$1$之间：若两个信号匹配，则幅度恰好为$1$；若根本不匹配，则幅度为$0$。

== 互相关的特性

- 若信号的长度为$n$个点，模式的长度为$m$个点，则互相关的长度为$n + m − 1$个点。请注意，有时较短的向量会用零填充到较长向量的长度，从而导致输出长度为$2 ∗ n − 1$。
- 若信号乘以因子$a$，特征乘以因子或$b$，则互相关函数的最大值将增加因子$a * b$。因此，若信号增加$a$倍，则自相关函数的最大值也会增加$a^2$倍。

== 互相关和卷积

信号$𝒙$与内核$𝒘$的卷积，相当于将权重为$𝒘$的 FIR 滤波器应用于信号$𝒙$。简单说就是，两个函数中一个函数经过反转和位移后再相乘得到的积的积分：

$
  (f ∗ g)(t) = ∫_(-∞)^(∞) f(τ) g(t - τ) dd(τ)
$

#figure(
  image("images/filter-conv.png", width: 40%),
  caption: "卷积运算",
  supplement: [图],
)

互相关是两个函数之间的滑动点积或滑动内积。互相关中的滤波器不经过反转，而是直接滑过函数$f$。$f$与$g$之间的交叉区域即是互相关。本质上是执行逐元素乘法和加法。

#figure(
  image("images/filter-conv-corr.png", width: 40%),
  caption: "卷积 vs 互相关 vs 自相关",
  supplement: [图],
)
||||||| 3bffbea
=======
#import "@preview/qooklet:0.2.0": *
#show: qooklet.with(
  title: "信号处理2",
  author: "SZZX",
  footer-cap: "SZZX",
  header-cap: "雷达基础知识手册",
  lang: "zh",
)

= 滤波器特性

== 脉冲和阶跃响应

- 脉冲响应：信号从0到1再到0的过程
- 阶跃响应：信号从0到1的过程

#figure(
  image("images/filter-response.png", width: 40%),
  caption: "脉冲和阶跃响应",
  supplement: "图",
)

=== 频率响应

LTI 传递函数的一个重要特性是，具有给定频率的正弦输入总是会产生具有相同频率的正弦输出，仅修改相位和/或幅度。传递增益的幅度和相位可以方便地表示为复数：幅度对应于复数的长度，相位对应于角度。这可用于表征滤波器响应。

#figure(
  image("images/filter-response-freq.png", width: 40%),
  caption: "频率响应特征",
  supplement: "图",
) <freqz>

@freqz 的 x 轴包含归一化频率，其中 1 对应于 Nyquist 频率。根据 Nyquist-Shannon 采样定理，信号中的信息可忠实再现的频率上限为

$ f_("Nyquist") = f_("sample") / 2 $

#figure(
  image("images/filter-response-nyq.png", width: 40%),
  caption: "采样再现",
  supplement: "图",
)

== 因果滤波器的伪影

因果滤波器的初始瞬态是由于滤波器窗口比已有的可用输入数据长引起的。按惯例，相应的缺失输入值设置为0，并且在数据分析中最好忽略瞬态。另一个影响是输出数据的延迟，这直接关系到数据的实时分析。

#definition[
  伪影（artifacts）：滤波器初始化值显示的“初始瞬态”，其输出相对输入总是存在时间延迟。
]

== 非因果滤波器的伪影

对离线分析，使用以居中分析窗口（centered analysis window）通常更方便。这消除了滤波后的输出信号$y$相对输入信号$x$的时间延迟问题。在 DSP 语言中，这称为非因果滤波器，因为未来点包含在$y_i$的评估中，这在数据的实时分析中是不可能的。

= 常用预处理技术

== 去趋势

去趋势是一种常见的预处理技术，用于消除数据中的趋势。趋势是数据中的长期变化，通常由数据中的周期性或线性变化引起。去趋势的目的是消除这种长期变化，以便更好地分析数据的短期变化。

#sgrid(
  figure(
    image("images/find-trend.png", width: 90%),
    caption: "趋势数据",
  ),
  figure(
    image("images/find-trend2.png", width: 90%),
    caption: "去趋势数据",
  ),
  columns: (200pt,) * 2,
  gutter: 2pt,
  caption: none,
)

```python
dt_ecgl = signal.detrend(
    ecgl["ecgl"], axis=-1, type="linear", bp=0, overwrite_data=False
)
dt_ecgnl = signal.detrend(
    ecgnl["ecgnl"], axis=-1, type="linear", bp=0, overwrite_data=False
)
```

可以看到，这些趋势已有效地去除，可以看到与原始信号相比，信号的基线已不再偏移，它们现在可用于进一步处理。

== 去干扰

=== Butterworth 陷波滤波器

Butterworth 滤波器旨在在通带中具有尽可能平坦的频率响应。因此，它也被称为“最大平坦幅度”滤波器。它可以用作低通、高通或带通滤波器。

这里的陷波宽度定义为 59 至 61 Hz 的频率区间。滤波器至少去除该范围内频率分量的一半功率。

```python
Fs = 1000
nom, denom = signal.iirfilter(
    N=5, Wn=[59, 61], btype="bandstop", analog=False, ftype="butter", fs=Fs
)
freq, h = signal.freqz(nom, denom, fs=Fs)
```

#figure(
  image("images/find-butterworth.png", width: 40%),
  caption: "Butterworth 滤波器",
  supplement: "图",
)

#warning[
  警告：请小心低滤波器频率的高阶（n ≥ 4）Butterworth 滤波器，其中，$[b,a]$语法可能会因舍入误差而导致数值问题。在这种情况下，应使用 IIR 滤波器推荐的“SOS”（second-order sections）语法或“ZPK”（zero-pole-gain）语法。
]

- 根据应用的不同，其他频率响应可能更可取。如，Chebyshev 滤波器提供比 Butterworth 滤波器更锐利的频率响应，而 Bessel 滤波器的优点是在时域中不会出现过冲（overshoot）。
- 为了将数据拟合到参数模型，使用原始数据几乎总是比使用预平滑数据更好，因为平滑已经丢弃了可用信息。

=== Filtfilt 滤波器

`Filtfilt` 是常见的非因果滤波器，其通过对数据运行滤波器两次来避免这种延迟：一次从前到后，一次从后到前。

```python
x_volt = openLoop["openLoopVoltage"]
t_volt = np.arange(len(x_volt)) / Fs
buttLoop = signal.filtfilt(nom, denom, x_volt.flatten())
```

#figure(
  image("images/find-filtfilter.png", width: 40%),
  caption: "Filtfilt 滤波器",
  supplement: "图",
)

#tip[
  要小心，滤波器的多次应用会改变滤波器特性。
]

== 重采样

若构造一个均匀加权的移动平均滤波器，它将去除相对于滤波器持续时间而言具有周期性的任何分量。以 1000 Hz 采样时，在 60 Hz 的完整周期内，大约有$1000\/60=16.667$个采样。尝试“向上舍入”并使用一个 17 点滤波器。这将在$1000\/17=58.82$Hz的基频下为我们提供最大滤波效果。

#figure(
  image("images/find-savgol.png", width: 40%),
  caption: "Savitzky-Golay 滤波器",
  supplement: "图",
)

然而，虽然电压明显经过平滑处理，但它仍然包含小的 60 Hz 波纹。若对信号进行重采样，以便通过移动平均滤波器捕获 60 Hz 信号的完整周期，就可以显著减弱该波纹。若以$17*60=1020$Hz对信号进行重采样，可以使用17点移动平均滤波器来去除 60 Hz 的电线噪声。

#figure(
  image("images/find-resample.png", width: 40%),
  caption: "重采样",
  supplement: "图",
)

== 去峰值

=== Hampel 滤波器

中值滤波器会去除尖峰，但同时也会去除原始信号的大量数据点。Hampel 滤波器的工作原理类似于中值滤波器，但它仅替换与局部中位数值相差几倍标准差的值。

```python
def hampel(x, k, n_sigma=3):
    """
    x: pandas series of values from which to remove outliers
    k: size of window/2
    n_sigma: number of standard deviations to use; 3 is standard
    """
    arraySize = len(x)
    idx = np.arange(arraySize)
    output_x = x.copy()
    output_idx = np.zeros_like(x)

    for i in range(arraySize):
        mask1 = np.where(idx >= (idx[i] - k), True, False)
        mask2 = np.where(idx <= (idx[i] + k), True, False)
        kernel = np.logical_and(mask1, mask2)
        median = np.median(x[kernel])
        std = 1.4826 * np.median(np.abs(x[kernel] - median))
        if np.abs(x[i] - median) > n_sigma * std:
            output_idx[i] = 1
            output_x[i] = median

    return output_x, output_idx.astype(bool)
```

#sgrid(
  figure(
    image("images/filter-med.png", width: 90%),
    caption: "中值滤波器",
  ),
  figure(
    image("images/filter-hampel.png", width: 90%),
    caption: "Hampel 滤波器",
  ),
  columns: (200pt,) * 2,
  gutter: 2pt,
  caption: none,
)

= 事件和特征查找

通常需要在数据流中查找特定位置。如，人们可能想知道信号何时超过给定阈值。或者，在分析运动信号时，人们可能想找出运动开始的点和结束的点。若数据是时间序列，这些位置通常称为事件（event）。

== 查找简单的特征

真实的测量数据总是包含噪声（并且通常比您想要的多得多）。为了从分析中消除噪声影响，从连续数据到二进制真/假表示通常会很有帮助。

例如，可以通过以下分析步骤来找到位置记录中的移动周期

+ 找到一个阈值
+ 对于每个数据点，计算信号是否高于阈值（真/假）。
+ 查找特征

== 插值

当找到曲线穿过给定阈值的点时，我们可能需要比数据中更高的精度。为此，我们可以通过插值找到记录样本之间的数据点。

=== 线性插值

最简单的插值形式是线性插值，其中样本之间的点是通过相邻样本之间的线性连接获得的。虽然这是一种计算速度快的方法，但它有两个缺点：

- 这不是很准确
- 所得曲线的一阶导数在样本位置处不连续

=== 三次样条插值

线性插值的缺点可以通过三次样条插值来克服。样条表明多项式系数满足两个条件：

- 它们在样本位置是连续的
- 每个多项式末尾的导数在一定阶数下是连续的

#figure(
  image("images/find-interpol.png", width: 40%),
  caption: "插值",
  supplement: "图",
)

#tip[
  插值样条与 B 样条明显不同：前者总是经过给定的数据点，而后者则被数据吸引。
]

= 测度

== 互相关

在信号处理中，互相关（cross-correlation）是对两个序列的相似性的度量，作为一个序列相对于另一个序列的位移的函数。这也称为滑动点积或滑动内积。它通常用于在长信号中搜索较短的已知特征。互相关本质上类似于两个函数的卷积。

为了计算互相关，较短的特征通常用零填充以获得与较长信号相同的长度$n$。按照这种约定，$x$和$y$的互相关函数可以通过以下方式获得

$
  R_(x y)(m) = cases(
    ∑_(i = 0)^(n - m - 1) x_(m + i) y_i^* quad & 0 ≤ m < n - 1,
    R_(y x)^* (-m) quad &1 - n < m < 0
  )
$ <cross_corr>

其中，$x$和$y$是长度为$n (n > 1)$的复向量，$y^*$是$y$的复共轭。式中互相关$R_(x y)$的定义不是固定的，并且可以根据$x$和$y$的序列以及输入值为复数时共轭的元素而变化。为了优化速度，通常不直接使用@cross_corr 来实现互相关，而是通过快速 Fourier 变换（Fast Fourier Transform，FFT）。

== 比较信号

让我们考虑如何评估两个信号的相似性，我们将其称为信号和特征。为了找到相似性，我们需要某种“相似性函数”，当特征与信号匹配时，该函数具有最大值，并且随着信号和特征之间的差异增加而减小。

#figure(
  image("images/find-crosscorr.png", width: 40%),
  caption: "互相关",
  supplement: "图",
) <cross_corr2>

将信号的一部分与特征进行比较所需要做的就是将信号的该部分与特征相乘。若想找出特征需要移动多少才能匹配信号，我们可以计算不同相对移动的相似度，并选择相似度最大的移动。对于与给定信号/特征范围之外的元素的乘法（@cross_corr2），相应的丢失数据通常被$0$替换。

#figure(
  table(
    columns: (auto,) * 9,
    inset: 0.3em,
    align: left + horizon,
    stroke: no-left-right(rgb("000")),
    [0], [0], [*2*], [*1*], [*4*], [*3*], [0], [0], [],
    [1], [3], [2], [], [], [], [], [], [$arrow.r 2^* 2 = 4$],
    [], [1], [3], [2], [], [], [], [], [$arrow.r 2^* 3 + 1^* 2 = 8$],
    [], [], [1], [3], [2], [], [], [], [$arrow.r 2^* 1 + 1^* 3 + 4^* 2= 13$],
    [], [], [], [1], [3], [2], [], [], [$arrow.r 1^* 1 + 4^* 3 + 3^* 2= 19$],
    [], [], [], [], [1], [3], [2], [], [$arrow.r 4^* 1 + 3^* 3 = 13$],
    [], [], [], [], [], [1], [3], [2], [$arrow.r 3^* 1 = 3$],
  ),
  caption: "互相关",
  supplement: [表],
  kind: table,
)

总而言之，互相关提供了两条信息：

- 信号和特征的相似度（互相关的最大值）
- 信号和特征之间的相对移动（互相关的最大值位置）

=== 自相关

若比较的两个信号相同，则结果称为自相关（auto-correlation）。当然，自相关函数不用于查找事件。相反，它可以用于检测信号中的周期性（periodicity）。在考虑平均偏移之后，自相关还用于检测信号中的能量，因为谐波信号中的能量与幅度的平方成正比。

在自相关（即信号与其自身的互相关）中，始终会在零滞后处出现峰值，其大小就是信号能量。

#figure(
  image("images/find-autocorr.png", width: 40%),
  caption: "自相关",
  supplement: "图",
)

=== 归一化

为了评估信号的形状（无论其总体持续时间、幅度或偏移量如何），我们必须对信号进行归一化。因此，归一化必须考虑信号的三个方面：

- 偏移（offset）：为了消除恒定偏移的影响，我们可以减去信号的平均值，从而避免三角形伪影，或者减去信号的最小值，从而确保互相关的输出始终为正；
- 持续（duration）：为了确保两个信号具有相同的长度，我们可以将它们插值到固定数量的数据点；
- 幅度：标准化互相关信号幅度的最常见方法是使自相关函数的最大值 = 1。这可以通过以下方式实现

$
  "sig"_("norm") = frac("sig"_("raw"), sqrt(max("autocorr"("sig"_("raw")))))
$ <sig_norm>

通过减去最小值和用方程归一化的幅度来消除偏移的信号。 @sig_norm 的幅度介于$0$和$1$之间：若两个信号匹配，则幅度恰好为$1$；若根本不匹配，则幅度为$0$。

== 互相关的特性

- 若信号的长度为$n$个点，模式的长度为$m$个点，则互相关的长度为$n + m − 1$个点。请注意，有时较短的向量会用零填充到较长向量的长度，从而导致输出长度为$2 ∗ n − 1$。
- 若信号乘以因子$a$，特征乘以因子或$b$，则互相关函数的最大值将增加因子$a * b$。因此，若信号增加$a$倍，则自相关函数的最大值也会增加$a^2$倍。

== 互相关和卷积

信号$𝒙$与内核$𝒘$的卷积，相当于将权重为$𝒘$的 FIR 滤波器应用于信号$𝒙$。简单说就是，两个函数中一个函数经过反转和位移后再相乘得到的积的积分：

$
  (f ∗ g)(t) = ∫_(-∞)^(∞) f(τ) g(t - τ) dd(τ)
$

#figure(
  image("images/filter-conv.png", width: 40%),
  caption: "卷积运算",
  supplement: "图",
)

互相关是两个函数之间的滑动点积或滑动内积。互相关中的滤波器不经过反转，而是直接滑过函数$f$。$f$与$g$之间的交叉区域即是互相关。本质上是执行逐元素乘法和加法。

#figure(
  image("images/filter-conv-corr.png", width: 40%),
  caption: "卷积 vs 互相关 vs 自相关",
  supplement: "图",
)
>>>>>>> 541da1d0404719384d9b514f3827ce9961804b3e
