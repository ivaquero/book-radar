#import "lib/sci-book.typ": *
#show: doc => conf(
  title: "波束形成",
  author: ("SZZX"),
  footer-cap: "SZZX",
  header-cap: "雷达基础知识手册",
  outline-on: false,
  doc,
)

= 相控阵

在天线理论中，相控阵（phased array）通常是指电子扫描阵列，一种由计算机控制的天线阵列。它可以产生一束无线电波，通过调整独立单元的相位，在不移动天线的情况下使发射信号指向不同的方向。

相控阵天线是同向性（isotropic）天线，向各个方向均匀辐射。影响阵列波束干涉的因素有

- 单元个数
- 单元位置
- 单元指向

== 组成

相控阵由以下部件组成：

#block(
  height: 6em,
  columns(3)[
    - 发射
      - 波形生成
      - 发射机
      - 发射阵列
    - 模型
      - 环境
      - 目标
      - 干扰
    - 接收
      - 接收阵列
      - 接收机
      - 信号处理
  ],
)

= 数字波束形成

对于无方向性的单个天线单元而言，其方向图是全向的。但，我们可以利用数字信号处理的方式，对每一个天线阵元通道做相位补偿（此处补偿的相位差是由天线阵元空间位置的差异导致每个阵元接收信号的波程差引起的），使得天线阵元对期望方向的信号做同相叠加，最大化该方向的接受增益，实现该方向的是波束形成。而相位补偿最常用的方式就是加权。

== 和差波束

相控阵雷达可以对每个发射或接收通道进行相位补偿以此获得目标点最大同相信号叠加，因此在相控阵雷达中都会存在移相器（也可称为加权，给每个输出信号引入相位加权）。将接收支路进行加权后相加则就形成了和波束通道。

波束和差原理简单来说就是对波束形成方向形成主瓣（目标方向需要在主瓣内），即和波束；同时在波束形成方向形成零陷，即差波束。通过和差波束比值得到某一确定的值，从而知道目标方向偏离波束中心的方向大小，最后确定目标位置。

差波束有两种形成方式。第一种方法，假设和波束指向$θ$，则在$θ + Δ θ$，$θ - Δ θ$两个方向分别形成波束，并相减，即可到到差波束。第二种方法，需要 Bayliss 窗函数（window function）。Bayliss 分布是一种典型的差分布，可以使得阵列左右单元的相位互相反相。

