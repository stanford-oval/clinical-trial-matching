# Icon set — clinical-trial-matching

Three marks on one construction: a 75×75 grid, a faceted-left / round-right lens,
slate constraint edges, ink nodes, one blue accent.

| file | use |
|---|---|
| umbrella-display.svg | the tool overall, 32px and up |
| umbrella-small.svg | below 32px (favicon, tab, 16–24px UI) |
| umbrella-mono-ink.svg / -mono-white.svg | one-colour surfaces, avatars, print |
| verdict-display.svg | VERDICT, 32px and up |
| verdict-small.svg | below 32px |
| verdict-mono-ink.svg / -mono-white.svg | one-colour surfaces |
| satir-display.svg | SAT·IR, unchanged from the original logo plus the blue cross |
| satir-mono-ink.svg | one-colour SAT·IR |

## Two cuts, not one drawing scaled

Display cuts carry 1.875-unit hairline edges and 2.25-radius nodes. Those vanish
under rasterisation, so the small cuts drop them, thicken what remains
(spokes 2.6, cross 4.2, route 3.8) and enlarge the VERDICT control node to 4.4.
Use the small cut below 32px. Monochrome always uses the small drawing: with the
accent gone, the hairlines have no means of separation.

## Colour

- ink `#0f172a`
- constraint edges `#94a3b8` (on dark: `#64748b`)
- accent `#2563eb`, gradient `#2563eb → #60a5fa` (on dark: `#60a5fa → #93c5fd`)

The gradient stops are an approximation of the original SAT·IR logo file, whose
ring gradient could not be read out of the PDF. Replace with the real stops when
available; they appear once per file in `<linearGradient id="g">`.

## Clear space and minimum size

Clear space: 0.25 × icon width on all sides. Minimum: 16px for the small cuts,
32px for display. Do not recolour the accent, do not add a tick, and do not
scale the display cut below 32px.

## VERDICT balance revision

The control node sits on the lens centre (33.75, 33.75) rather than left of and
below it. The stem climbs from the lower-left facet in two steps to reach it,
and the route leaves at y=24.5 through a 26° ring gap centred at -21°. This
brings the drawn content's centre onto the lens axis; the earlier draft centred
6.3 units off it.

## Umbrella revision — "converge and leave"

The umbrella no longer derives from SAT·IR alone. Three constraint edges run in
from the left facet nodes to a hollow pivot on the lens centre, and a single
blue arrow leaves due right through a 26° ring gap centred at 0°. The fan and
ink nodes come from SAT·IR, the pivot and exiting arrow from VERDICT; the
clinical cross is dropped, since the matching is now carried by the convergence.

Small cut keeps two fan edges at 2.6, drops the hairline third edge and the
nodes, and takes the pivot to r4.6.
