#import "lib/lib.typ": *
#show: chapter-style.with(
  title: "相关工具箱 API",
  info: info,
)

= 雷达

- APP
  - radarDesigner
  - pulseWaveformAnalyzer

= 汽车

== 场景

#block(
  height: 9em,
  columns()[
    - 创建对象
      - `scenario = drivingScenario`
    - 动作
      - `isRunning = advance(scenario)`
      - `rec = record(scenario)`
    - 动作（无返回值）
      - `restart(scenario)`
  ],
)

== 道路

#block(
  height: 9em,
  columns()[
    - 添加对象
      - `road(scenario,roadcenters)`
      - `roadNetwork(scenario,'OpenDRIVE',filename)`
      - `roadGroup(scenario,rg)`
    - 获取属性
      - `rbdry = roadBoundaries(scenario)`
      - `rdMesh = roadMesh(ac)`
  ],
)

== 参与者

#block(
  height: 9em,
  columns()[
    - 添加对象
      - `ac = actor(scenario)`
      - `vc = vehicle(scenario)`
      - `barrier(scenario,rd)`
    - 添加属性
      - `trajectory(ac,waypoints)`
      - `smoothTrajectory(ac,waypoints)`
    - 获取属性
      - `poses = actorPoses(scenario)`
      - `poses = targetPoses(ac)`
  ],
)

== 绘图

=== 对象

#block(
  height: 9em,
  columns()[
    - `chasePlot(ac)`
  ],
)

== APP

drivingScenarioDesigner
