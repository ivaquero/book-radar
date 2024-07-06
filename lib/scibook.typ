// multi-languages
#import "@preview/linguify:0.4.1": *
// indent
#import "@preview/indenta:0.0.3": fix-indent
// header-footer
#import "@preview/hydra:0.5.0": *
// chemistry
#import "@preview/whalogen:0.2.0": ce
// physics
#import "@preview/physica:0.9.2": *
// theorems
#import "@preview/ctheorems:1.1.2": *
// banners
#import "@preview/gentle-clues:0.9.0": *
// figures
#import "@preview/subpar:0.1.1": grid as sgrid
// wrap
#import "@preview/wrap-it:0.1.0": wrap-content
// diagram
#import "@preview/fletcher:0.5.0": diagram, node, edge
// numbering
#import "@preview/i-figured:0.2.4"
// index
#import "@preview/in-dexter:0.4.2": *

#let conf(
  title: none,
  author: (),
  header-cap: [],
  footer-cap: [],
  outline-on: true,
  eqnumstyle: "1",
  eqnumsep: ".",
  eqnumlevel: 1,
  par-leading: 1em,
  list-indent: 1.2em,
  block-above: 1em,
  block-below: 0.5em,
  figure-break: false,
  lang: "zh",
  doc,
) = {
  set page(
    paper: "a4",
    numbering: "1",
    header: context {
      set text(size: 8pt)
      if calc.odd(here().page()) {
        align(right, [#header-cap #h(6fr) #emph(hydra(1))])
      } else {
        align(left, [#emph(hydra(1)) #h(6fr) #header-cap])
      }
      line(length: 100%)
    },
    footer: context {
      set text(size: 8pt)
      let page_num = here().page()
      if calc.odd(page_num) {
        align(
          right,
          [#footer-cap #datetime.today().display("[year]-[month]-[day]") #h(6fr) #page_num],
        )
      } else {
        align(
          left,
          [#page_num #h(6fr) #footer-cap #datetime.today().display("[year]-[month]-[day]")],
        )
      }
    },
  )
  set heading(numbering: "1.1")

  set par(
    first-line-indent: 2em,
    justify: true,
    leading: par-leading,
    linebreaks: "optimized",
  )
  set block(above: block-above, below: block-below)
  set list(indent: list-indent)
  set enum(indent: list-indent)

  let font = {
    if lang == "en" {
      "Arial"
    }
    if lang == "zh" {
      "Songti SC"
    }
  }

  set text(
    font: font,
    size: 10.5pt,
    lang: lang,
  )

  let lang_data = toml("lang.toml")
  set-database(lang_data)

  set ref(supplement: it => {
    if it.func() == heading {
      linguify("chapter")
    } else if it.func() == table {
      it.caption
    } else if it.func() == image {
      it.caption
    } else if it.func() == figure {
      it.supplement
    } else if it.func() == math.equation {
      linguify("eq")
    } else { }
  })

  show heading: i-figured.reset-counters.with(level: 2)
  show math.equation: i-figured.show-equation

  set figure.caption(separator: "  ")

  show figure: it => align(
    center,
    block(breakable: figure-break)[
      #it.body#it.caption
    ],
  )

  show raw.where(block: true): block.with(
    fill: luma(240),
    inset: .8em,
    radius: 5pt,
    width: 100%,
  )

  if outline-on == true [
    #outline(
      title: linguify("content"),
      indent: auto,
      depth: 2,
    )
    #pagebreak()
  ]

  show link: underline
  show: thmrules
  show: fix-indent()
  doc
}

// tables
#let frame(stroke) = (x, y) => (
  top: if y < 2 {
    stroke
  } else {
    0pt
  },
  bottom: stroke,
)

#let ktable(data, k, inset: 0.3em) = table(
  columns: k,
  inset: inset,
  align: center + horizon,
  stroke: frame(rgb("000")),
  ..data.flatten(),
)

// codes
#let code(text, lang: "python", breakable: true, width: 100%) = block(
  fill: rgb("#f3f3f3"),
  stroke: rgb("#dbdbdb"),
  inset: (x: 1em, y: 1em),
  outset: -.3em,
  radius: 5pt,
  spacing: 1em,
  breakable: breakable,
  width: width,
  raw(
    text,
    lang: lang,
    align: left,
    block: true,
  ),
)

// theorems
#let definition = thmbox(
  "definition",
  linguify("definition"),
  base_level: 1,
  separator: [#h(0.5em)],
  padding: (top: 0em, bottom: 0em),
  fill: rgb("#ffffff"),
  // stroke: rgb("#000000"),
  inset: (left: 0em, right: 0.5em, top: 0.2em, bottom: 0.2em)
)

#let theorem = thmbox(
  "theorem",
  linguify("theorem"),
  base_level: 1,
  separator: [#h(0.5em)],
  padding: (top: 0em, bottom: 0.2em),
  fill: rgb("#e5eefc"),
  // stroke: rgb("#000000")
)

#let lemma = thmbox(
  "theorem",
  linguify("lemma"),
  separator: [#h(0.5em)],
  fill: rgb("#efe6ff"),
  titlefmt: strong,
)

#let corollary = thmbox(
  "corollary",
  linguify("corollary"),
  // base: "theorem",
  separator: [#h(0.5em)],
  titlefmt: strong
)

#let algo = thmbox(
  "",
  linguify("algorithm"),
  base_level: 1,
  separator: [#h(0.5em)],
  padding: (top: 0em, bottom: 0.2em),
  fill: rgb("#faf2fb"),
  titlefmt: strong,
)

// banners
#let tip(title: linguify("tip"), icon: emoji.lightbulb, ..args) = clue(
  accent-color: yellow,
  title: title,
  icon: icon,
  ..args,
)

#let alert(title: linguify("alert"), icon: emoji.excl, ..args) = clue(
  accent-color: red,
  title: title,
  icon: icon,
  ..args,
)
