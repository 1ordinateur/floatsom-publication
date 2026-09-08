# Publication SVG styling

The SVGs in this directory contain manual artwork edits and are the source of
the current TMLR figure PDFs. Edit these SVGs directly and export their PDFs;
do not overwrite them by rerunning the benchmark figure generators.

The September 2026 styling pass uses Figure 7 as the reference. Figures 1 and 2
were excluded. Existing panel arrangements, data coordinates, confidence-interval
endpoints, significance annotations, and embedded observation images were retained.

- Forest plots share red/grey observations, diamond summaries, interval strokes,
  and a common legend. S1 retains its distinct Wilcoxon interval description.
- Font sizes are normalized to printed width, including existing outlined text.
  Legends, panel letters, panel headings, and explanatory subtitles were enlarged
  by 30% in a subsequent readability pass. At Figure 7's 3269.87-unit width,
  legend text now uses approximately 47.23 units and panel headings 52 units
  (6.76 and 7.44 pt at the manuscript's 6.5-inch width). Axis and dataset labels
  remain approximately 36.33 units (5.2 pt); overall figure titles were subsequently enlarged by 30% (65 effective units
  at Figure 7's width).
  Outlined legend and panel text was enlarged directly as complete vector groups.
  Legend entries were re-spaced, and canvas heights expanded where necessary,
  retaining the same printed figure widths and original plot geometry.
- Figure 11C's heading identifies stability, matching its underlying analysis.
- Figure 13 and S4 use matching GPU colors and symbols for 1, 2, 3, 4, and 8 GPUs.
  Their LaTeX captions list the same five counts. GPU legend labels are 7.8 pt at
  manuscript width, with a centered shading key and 4 mm bottom padding.
- S3's canvas includes its previously clipped bottom ticks and legend. Its unused
  right margin was removed, and a shared QE axis label was added.
- Figure 11's legend uses the forest legend's text/marker proportions with 7 mm
  gaps between complete entries. Figures 12 and 14 use spaced topology blocks.
- Figure 15 reserves 4 mm between the QE label and forest legend, 6 mm before
  the D heading, and 3 mm between that heading and the runtime axes. The runtime
  artwork is translated as a unit, without scaling, to preserve these gaps.
  Its separate runtime legend uses the common symbol proportions and spacing.
- Titled figures use Figure 5's title-to-content gap: approximately 5.56 mm at
  manuscript width. All artwork below each title is translated together, with
  the canvas height extended by the same amount. Existing panel, axis and legend
  spacing is preserved. For S3 and S4, the first content is an explanatory
  subtitle. Figures 1 and 2 have no embedded figure title.
  All 17 embedded titles, including Figure 5 and supplementary titles, now use
  the additional 30% enlargement. Their top-left positions and the established
  title-to-content gaps are retained, with artwork translated down only enough
  to accommodate the taller titles.
- The manuscript uses `\raggedbottom` to keep unused height at the bottom of a
  text page instead of expanding caption-to-text gaps.

PDFs were exported from the edited vectors with CairoSVG and Arial. The accepted
TMLR draft, with restored authors and acknowledgments, was rebuilt with Tectonic
and contains 56 pages. Publication date and OpenReview metadata await author
input. Figure exports and affected manuscript layouts were inspected.
Checks confirmed preservation of original panel IDs, data path instructions,
and named data-path centers. Earlier manuscript asset directories and plotting
generators were not changed.
