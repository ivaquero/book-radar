<<<<<<< HEAD
#import "@local/scibook:0.1.0": *
#show: doc => conf(
||||||| 3bffbea
#import "lib/sci-book.typ": *
#show: doc => conf(
=======
#import "@preview/qooklet:0.2.0": *
#show: qooklet.with(
>>>>>>> 541da1d0404719384d9b514f3827ce9961804b3e
  title: "时间配准",
  author: "SZZX",
  footer-cap: "SZZX",
  header-cap: "雷达基础知识手册",
  lang: "zh",
)

= 时间配准
<时间配准>

时间配准（time registration），又称时间同步，是将多个量测单元经时间对准后，剩余的时间偏差控制在容许的范围内的处理过程。包括绝对配准和相对配准，分别对应与天文时间和高精度主时钟的同步。

数据融合过程中，由于不同设备的数据链不同，所以融合前必须进行时间配准。

= 常用时间

#definition[
  儒略日（Julia Day，JD）是指从公元前 4713 年 1 月 1 日 12 时开始连续计算得出的天数（不满一日的部分用小数表示）。儒略日中的天数被称为儒略日数（JDN）。
]

#warning[
  儒略日 ≠ 儒略历。

  前者是法国学者 Joseph Justus Scaliger（1540～1609）设计的一种历法，后者是由罗马共和国独裁官 Gaius Iulius Caesar 在公元前45年1月1日颁布的历法。
]

#definition[
  协调世界时（UTC），其以原子时的秒长为基础，把时间分为天、小时、分钟和秒。通常，天是使用 Gregorian 历（公历）定义的，但也能使用儒略日。
]
