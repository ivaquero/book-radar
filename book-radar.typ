#import "lib/lib.typ": *

#cover(info)
#contents(depth: 1, info: info)

#let chapter(filename) = {
  include filename
  context counter(heading).update(0)
}

#chapter("01-雷达方程.typ")
#chapter("02-波束形成.typ")
#chapter("03-信号处理.typ")
#chapter("04-信号处理2.typ")
#chapter("05-信号处理3.typ")
#chapter("06-空间配准.typ")
#chapter("07-时间配准.typ")
#chapter("matlab-radar.typ")
