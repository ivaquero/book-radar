#import "lib/lib.typ": *
#show: chapter-style.with(title: "雷达与自动驾驶工具箱 API", info: info)

本附录 API 清单以 MATLAB R2025a 为基准。各命令的权威说明见 #link("https://ww2.mathworks.cn/help/releases/R2025a/radar/index.html")[Radar Toolbox] 与 #link("https://ww2.mathworks.cn/help/releases/R2025a/driving/index.html")[Automated Driving Toolbox] 文档；函数随版本增删，核对时以上述页面为准。

= 雷达

- APP
  - radarDesigner
  - pulseWaveformAnalyzer

= 汽车

== 场景

#columns()[
  #[
    - 创建对象
      - `scenario = drivingScenario`
    - 动作
      - `isRunning = advance(scenario)`
      - `rec = record(scenario)`
  ]
  #colbreak()

  #[
    - 动作（无返回值）
      - `restart(scenario)`
  ]
]

== 道路

#columns()[
  #[
    - 添加对象
      - `road(scenario,roadcenters)`
      - `roadNetwork(scenario,'OpenDRIVE',filename)`
      - `roadGroup(scenario,rg)`
  ]
  #colbreak()

  #[
    - 获取属性
      - `rbdry = roadBoundaries(scenario)`
      - `rdMesh = roadMesh(ac)`
  ]
]

== 参与者

#columns()[
  #[
    - 添加对象
      - `ac = actor(scenario)`
      - `vc = vehicle(scenario)`
      - `barrier(scenario,rd)`
  ]
  #colbreak()

  #[
    - 添加属性
      - `trajectory(ac,waypoints)`
      - `smoothTrajectory(ac,waypoints)`
    - 获取属性
      - `poses = actorPoses(scenario)`
      - `poses = targetPoses(ac)`
  ]
]

== 绘图

=== 对象

#columns()[
  #[
    - `chasePlot(ac)`
  ]
]

== APP

drivingScenarioDesigner
