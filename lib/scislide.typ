// slides
#import "@preview/touying:0.4.2": *
// indent
#import "@preview/indenta:0.0.3": fix-indent
// physics
#import "@preview/physica:0.9.2": *
// theorems
#import "@preview/ctheorems:1.1.2": *
// banners
#import "@preview/gentle-clues:0.9.0": *
// figures
#import "@preview/subpar:0.1.1": grid as sgrid

#let conf(
  title: none,
  subtilte: none,
  author: (),
  institute: none,
  footer: [],
  outline-title: [内容提要],
  ending: [感谢各位老师聆听],
  doc,
) = {
  set text(font: "Songti SC", weight: "light", size: 20pt)
  set par(
    first-line-indent: 2em,
    justify: true,
    leading: 1em,
    linebreaks: "optimized",
  )
  set block(above: 1em, below: 0.5em)
  set math.equation(numbering: "(1)")
  set ref(supplement: it => {
    if it.func() == table {
      it.caption
    } else if it.func() == image {
      it.caption
    } else if it.func() == figure {
      it.supplement
    } else if it.func() == math.equation { } else { }
  })

  let s = themes.metropolis.register(
    aspect-ratio: "16-9",
    footer: footer,
  )

  let s = (s.methods.info)(
    self: s,
    title: text(title, size: 40pt),
    subtitle: subtilte,
    author: [#text(author, size: 18pt, font: "Kaiti SC")],
    date: datetime.today(),
    institution: institute,
  )

  let s = (s.methods.enable-transparent-cover)(self: s)

  let s = (s.methods.colors)(
    self: s,
    neutral-lightest: rgb("#ffffff"),
    primary-dark: rgb("#3297df"),
    secondary-light: rgb("#ff0000"),
    secondary-lighter: rgb("#fcbd00"),
  )

  let (init, slides, alert, speaker-note) = utils.methods(s)
  show: init
  show strong: alert
  show link: underline
  show: thmrules
  show: fix-indent()

  let (slide, empty-slide, title-slide, new-section-slide, focus-slide) = utils.slides(s)
  show: slides.with(outline-title: outline-title)

  doc

  focus-slide[
    #text(ending, font: "Kaiti SC", size: 40pt)
  ]
}

#let tip(title: "", icon: emoji.lightbulb, ..args) = clue(
  accent-color: rgb("#ffe66b"),
  title: title,
  icon: icon,
  ..args,
)
