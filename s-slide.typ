#import "lib/scislide.typ": *
#show: doc => conf(
  title: "一维 Kalman 滤波",
  author: ("试讲人：XXX"),
  footer: ("xxx学院"),
  doc,
)

= 铺垫与术语

== 系统状态

若一个模型可以被视为一个系统，那么根据其观测值（输入）的性质，可被划分为以下两类

- 静态系统（static system）：当前的输入与时间无关。如短时间内，一块金条的重量，一栋建筑的高度
- 动态系统（dynamic system）：当前的输出与时间有关，取决于系统过去时刻的状态（输入和输出）

== 状态空间模型

为了描述一个动态系统，我们往往需要建立一个动态模型（dynamic model），如使用雷达对一架飞机的位置进行测距，可得

$
  cases(
    x = x_0 + v_(x 0) Δ t + 1/2 a_x Δ t^2\
    y = y_0 + v_(y 0) Δ t + 1/2 a_y Δ t^2\
    z = z_0 + v_(z 0) Δ t + 1/2 a_z Δ t^2
  )
$

其中，参数$[x, y, z, v_x, v_y, v_z, a_x, a_y, a_z]$被称系统状态（system state）。故这样的模型又被称为状态空间模型（state space model）。

== 状态更新方程


对一个人的身高，进行多次量测，可写成递归形式

$
  hat(x)_(k)
  &= 1 / k (z_(1) + z_(2) + … + z_(k-1) + z_(k))\
  &= 1 / k (∑_(i=1)^(k-1) z_(i) + z_(k))\
  &= (k-1) / k 1 / (k-1) ∑_(i=1)^(k-1) z_(i) + 1 / (k-1) z_(k)\
  &= (k-1) / k hat(x)_(k|k-1) + 1 / k z_(k)\
  &= hat(x)_(k|k-1) + 1 / k (z_(k) - hat(x)_(k|k-1))
$

此方程被称为状态更新方程（state update equation）。于是，状态更新方程可描述为

#align(center, text("当前估计 = 预测 + 增益 × 新息", font: "Kaiti SC", size: 20pt))

#h(0em) 需要强调的是，估计值，又称滤波值，永远介于预测值和量测值之间，可以看着是一种利用加权平均的对量测进行平滑的手段。而这里的新息（innovation）即当前测量的残差，有时会使用$y$表示，即

$ y_(k) = z_(k) - hat(x)_(k|k-1) $

不难看出，随着量测次数$k$的增大，当前估计会越来越接近先验估计，这里即前$k$次量测的均值。在实际应用中，增益因子$1/k$往往用更一般的$K_(k) ∈ [0,1]$代替，$K$即是 Kalman 增益。为了不显得那么急功近利，我们暂时用$α_(k)$ 来表示

$ hat(x)_(k) = hat(x)_(k|k-1) + α_(k) (z_(k) - hat(x)_(k|k-1)) $

= 匀速模型

== 状态外推方程

对一个匀速（constant velocity，CV）航行的飞行器，雷达以恒定频率向目标发射追踪波束，周期为$Δ t$，则飞行器距离雷达的距离$x$可表示为

$
  hat(x)_(k) &= hat(x)_(k-1) + Δ t hat(dot(x))_(k-1)\
  hat(dot(x))_(k) &= hat(dot(x))_(k-1)
$

上述方程被称为状态外推方程（state extrapolation equation）或预测方程。又由于动态（时变）过程必然带来噪声，即过程噪声（process noise）。引入过程噪声后，上述两式表示为

$
  hat(x)_(k) &= hat(x)_(k-1) + Δ t hat(dot(x))_(k-1) \
  hat(dot(x))_(k) &= hat(dot(x))_(k-1)
$

== α-β 滤波

由于仅依赖状态外推方程并不能得到准确的$hat(x)$，此时还需要位置和速度的状态更新方程，分别对$hat(x)$和$hat(dot(x))$进行校正，即

$
  hat(x)_(k) &= hat(x)_(k|k-1) + α_(k) (z_(k) - hat(x)_(k|k-1))\
  hat(dot(x))_(k) &= hat(dot(x))_(k-1) + β (frac(z_(k) - hat(x)_(k|k-1), Δ t)\)
$

此即是 α-β 滤波，又称 g-h 滤波。

#sgrid(
  figure(
    image("images/gh-cv.png", width: 180%),
    caption: "",
    supplement: "图",
  ),
  figure(
    image("images/gaussian-prod.png", width: 150%),
    caption: "",
    supplement: [图],
  ),
  columns: (200pt,) * 2,
  gutter: 180pt,
  caption: "量测、预测与滤波",
  supplement: "图",
)

== 噪声与增益

由于采样间隔$Δ t$相对于目标跟踪的时间往往很小，故每个采样周期内的$v_x$可近似常数。若再假设过程噪声在各个采样周期之间相互独立，当过程噪声为白噪声（white noise）时，定义机动目标指数$λ$为

$ λ = frac(Δ t σ_v, σ_w) $

其中，$σ_v$和$σ_w$分别为过程噪声和量测噪声（measurement noise）的标准差。前者又称模型噪声，表示动态模型的不确定性，后者表示测定方法的不确定性。可得常系数滤波增益为

$
  α = -frac(λ^2 + 8λ- (λ+4)sqrt(λ^2+8λ), 8)\
  β = frac(λ^2 + 4λ - λ sqrt(λ^2+8λ), 4)
$

通常，$σ_w$是已知的，而$σ_v$较难获得，当后者误差较大时，α-β 滤波会失效。工程上常绕过$σ_v$直接采用如下经验值

$
  α = frac(2(2k-1), k(k-1))\
  β = frac(6, k(k+1))
$

其中，$α$和$β$分别从$k=1$和$k=2$开始计算，滤波从$k=3$开始工作。

#text(
  "Bar-Shalom Y, Fortmann T E, Cable P G. Tracking and Data Association. Academic Press Professional",
  style: "italic",
  font: "Times New Roman",
  size: 16pt,
)

#align(
  center,
  stack(
    dir: ltr,
    spacing: 2em,
    figure(
      image("images/gh-g.png", width: 50%),
      caption: "变化的 α（g）",
      supplement: "图",
    ),
    figure(
      image("images/gh-h.png", width: 50%),
      caption: "变化的 β（h）",
      supplement: "图",
    ),
  ),
)

由上不难看出，$α$控制着滤波的精度，$β$控制着滤波响应时间，后者与雷达精度呈正相关。

= 匀加速模型
<匀加速模型>

== α-β-γ 滤波

对一个匀加速（constant acceleration，CA）航行的飞行器，类比 CV 模型，有

$
  hat(x)_(k|k-1)
  &= hat(x)_(k) + hat(dot(x))_(k) Δ t + hat(dot.double(x))_(k) frac(Δ t^2, 2)\
  hat(dot(x))_(k|k-1)
  &= hat(dot(x))_(k) + hat(dot.double(x))_(k) Δ t\
  hat(dot.double(x))_(k|k-1)
  &= hat(dot.double(x))_(k)
$

化为递归形式，可得

$
  hat(x)_(k)
  &= hat(x)_(k|k-1) + α (z_(k) - hat(x)_(k|k-1)\)\
  hat(dot(x))_(k)
  &= hat(dot(x))_(k-1) + β (frac(z_(k) - hat(x)_(k|k-1), Δ t)\)\
  hat(dot.double(x))_(k)
  &= hat(dot.double(x))_(k-1) + γ (frac(z_(k) - hat(x)_(k|k-1), 0.5 Δ t^2)\)
$

此即是 α-β-γ 滤波，又称 g-h-k 滤波。其中，其增益和机动目标指数的关系为

$
  frac(γ^2, 4(1-α)) = λ^2\
  β = 2(2-α) - 4 sqrt(1-α) \
  γ = β^2 / α
$

类似α-β 滤波，α-β-γ 滤波的增益也有经验值，这里暂略去不表。

= Kalman 滤波模型

== 状态预测

与 α-β-γ 滤波相似，Kalman 滤波也基于数据融合的思想，但是一种变增益滤波。其在前者的基础上，增加了利用协方差校正步骤，即引入了协方差外推方程（又称估计不确定性外推方程）。

根据匀速运动模型，有

$
  hat(x)_(k) &= hat(x)_(k-1) + Δ t hat(dot(x))_(k-1) \
  hat(dot(x))_(k-1) &= hat(dot(x))_(k-1)
$

其协方差外推方程为

$
  p_(k|k-1)^x &= p_(k-1)^x + Δ t^2 p_(k-1)^v \
  p_(k|k-1)^v &= p_(k-1)^v + q_k
$

其中，$p^x$和$p^v$分别是位置和速度的协方差，$q_k$过程噪声方差。

== 状态校正

=== 状态更新方程

为了估计系统的当前状态，我们结合 2 个随机变量，先验估计$hat(x)_(k|k-1)$和量测$z$。Kalman 滤波将先前的状态估计与量测相结合，以最大程度地减少当前状态估计的不确定性。当前状态估计是量测的加权平均值和先前的状态估计，即

$
  & hat(x)_(k) = w_1 z_k + w_2 hat(x)_(k|k-1)\
  & w_1 + w_2 = 1
$

其中，$w_1$和$w_2$分别为量测和先验的权重。于是有

$ hat(x)_(k) = w_1 z_k + (1 - w_1) hat(x)_(k|k-1) $

协方差关系为

$ p_(k) = w_1^2 r_k + (1 - w_1)^2 p_(k|k-1) $ <cov_up>

#h(0em) 这里约定

- $p_(k)$：融合估计协方差，即最优协方差
- $p_(k|k-1)$：先验$hat(x)_(k|k-1)$的协方差
- $r_k$：量测$z_k$的协方差


对$p_(k)$求导

$
  frac(d p_(k|k), d w_1) = 2 w_1 r_k - 2 (1 - w_1) p_(k|k-1) = 0
$

#pagebreak()

得

$
  w_1 = frac(p_(k|k-1), p_(k|k-1) + r_k)
$

其中，$w_1$即为 Kalman 增益，通常用$K ∈ [0,1]$表示。其可描述为对预测值采纳的占比，即

$ K = frac("Var"(x), "Var"(x) + "Var"(z)) $

综上，Kalman 滤波的状态更新方程为

$ hat(x)_(k) = hat(x)_(k|k-1) + K_(k) (z_(k) - hat(x)_(k|k-1)) $ <st_up>

#pagebreak()

=== 协方差更新方程

将式@st_up 代入式@cov_up 可得

$
  p_(k) &= K_k^2 r_k + (1 - K_k)^2 p_(k|k-1)\
  &= (1 - K_k)p_(k|k-1)
$

此即为 Kalman 滤波的协方差更新方程。由于$(1 - K_k) ≤ 1$，可知

- 当量测不确定性高时，估计不确定性会不断减小，估计不确定性的收敛性会很慢
- 当量测不确定性较低时，Kalman 增益很高，估计不确定性将迅速趋于$0$。

== Kalman 增益

#h(1em) 重写式@st_up，得

$
  hat(x)_(k) = hat(x)_(k|k-1) + K_k (z_n - hat(x)_(k|k-1)) = (1 - K_k) hat(x)_(k|k-1) + K_k z_n
$

#h(1em) 不难看出，Kalman 增益$K_k$代表量测权重，$1 - K_k$项代表当前预测权重。

- 当量测不确定性高且估计不确定性低时，$K_k → 0$。故，我们给估计的权重大，给量测的权重小。
- 当量测不确定性较且估计不确定性高时，$K_k → 1$。故，我们给估计的权重小，给量测的权重大。

#tip[
  Kalman 增益在形成新估计时定义了量测权重和先验估计权重。它告诉我们，量测对估计的影响。
]

= Kalman 滤波流程

#block(
  height: 14.5em,
  width: 90%,
  columns(3)[
    - 初始化
      + 获取
        - 初始估计状态：$hat(x)_0$
        - 初始状态方差：$p_0$
        - 量测：$z_k$
        - 量测方差：$r_k$
      + 状态预测
        - 预测：$hat(x)_(k|k-1)$
        - 预测方差：$p_(k|k-1)$

    - 状态更新
      + 获取
        - 量测及其方差：$z_k$
        - 量测方差：$r_k$
        - 预测：$hat(x)_(k|k-1)$
        - 预测方差：$p_(k|k-1)$
      + 状态预测
        - 估计：$hat(x)_k$
        - 估计方差：$p_k$

    #image("images/kalman.drawio.png", width: 160%)
  ],
)


