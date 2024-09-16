// numbering
#import "@preview/i-figured:0.2.4"

#let chapter(filename) = {
  include filename
  context counter(heading).update(0)
}

#chapter("01-雷达方程.typ")
#chapter("02-波束形成.typ")
#chapter("03-信号处理.typ")
#chapter("04-空间配准.typ")
#chapter("05-时间配准.typ")
