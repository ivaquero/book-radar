#import "lib/sci-book.typ": *
#show: doc => conf(
  title: "汽车工具箱 API",
  author: ("数字技术研发中心"),
  footer-cap: "数字技术研发中心",
  header-cap: "雷达基础知识手册",
  outline-on: false,
  doc,
)

= 基础

== 场景

#block(
  height: 9em,
  columns()[
    - 创建
      - `scenario = drivingScenario`
    - 动作
      - `isRunning = advance(scenario)`
      - `rec = record(scenario)`
      - `restart(scenario)`
  ],
)

== 道路

#block(
  height: 9em,
  columns()[

  ],
)

== 车辆

#block(
  height: 9em,
  columns()[

  ],
)

== 参与者

#block(
  height: 9em,
  columns()[

  ],
)


= APP

drivingScenarioDesigner
