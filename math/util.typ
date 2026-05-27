// Shared style and helpers for Empo math documents.

#let setup(body) = {
  set page(fill: rgb("#1e1e1e"), height: auto)
  set text(lang: "en", fill: rgb("#e0e0e0"), size: 14pt)
  set math.equation(numbering: "(1)")
  body
}

#let note(body) = block(
  fill: rgb("#2a2a2a"),
  inset: 12pt,
  radius: 2pt,
  width: 100%,
  [*Note:* #body],
)

#let unnumbered(body) = [
  #set math.equation(numbering: none)
  #body
]

#let titled(name, body) = block(
  spacing: 1.5em,
  breakable: false,
  {
    block(below: 0.5em,
      text(size: 1em, fill: rgb("#a0a0a0"), style: "italic", name + ":"))
    body
  },
)
