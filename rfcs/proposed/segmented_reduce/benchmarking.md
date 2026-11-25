# Benchmarking Offset-Based Segmented Reduction on Real and Synthetic Segment Distributions

Table of contents:

- [Introduction](#introduction)
- [Review of Data Generation Approaches](#review-of-data-generation-approaches)
  - [Uniform lengths with a swept average (ModernGPU)](#uniform-lengths-with-a-swept-average-moderngpu)
  - [Fixed and bounded-uniform lengths (CUB)](#fixed-and-bounded-uniform-lengths-cub)
  - [Real sparse matrices (SpMV literature)](#real-sparse-matrices-spmv-literature)
  - [Real graphs (SNAP)](#real-graphs-snap)
  - [Real workloads in the CUB benchmarks](#real-workloads-in-the-cub-benchmarks)
  - [Synthetic power-law graphs (R-MAT, Graph500)](#synthetic-power-law-graphs-r-mat-graph500)
  - [Summary](#summary)
  - [Data sources and their users](#data-sources-and-their-users)
  - [Statistics of the real data sets](#statistics-of-the-real-data-sets)
  - [How the libraries measure segmented reduction](#how-the-libraries-measure-segmented-reduction)
- [Proposal](#proposal)
  - [Input regimes](#input-regimes)
  - [File format of the segments](#file-format-of-the-segments)
  - [Preparation of the data, step by step](#preparation-of-the-data-step-by-step)
  - [Delivery of the data](#delivery-of-the-data)
  - [Metrics and reporting](#metrics-and-reporting)
  - [What the time of a call consists of](#what-the-time-of-a-call-consists-of)
  - [Normalization of the times across devices](#normalization-of-the-times-across-devices)
  - [Prototype](#prototype)
- [Open Questions](#open-questions)
- [References](#references)

## Introduction

The [Offset-Based Segmented Reduction](README.md) RFC ([#2848](https://github.com/uxlfoundation/oneDPL/pull/2848)) adds
a `reduce_by_segment` overload whose segments are defined by `num_segments + 1` offsets. Its performance depends on the
distribution of the segment lengths much more than on the input size alone: the work per segment ranges from nothing to
the whole input, which makes the work hard to distribute evenly. The hard cases are many short or empty segments, each
of which adds work of its own, and skewed distributions, where a few very long segments coexist with many short ones and
make the work hard to balance. The offsets may lie outside the input, and offsets that decrease are erroneous but safe
(see [Empty, Overlapping, Negatively Sized, and Unordered
Segments](README.md#empty-overlapping-negatively-sized-and-unordered-segments) in the RFC): the implementation clamps
every offset to `[0, input size]`, so a segment outside the input or with its end before its begin reduces nothing and
gets the initial value, like an empty segment. Such segments therefore need no regime of their own.

The benchmarks used so far generate the segments synthetically: the lengths are drawn uniformly from one or two ranges
(e.g. 1..1024, 1..16, 2K..128K, 64K..256K, and a mixture of 95 % short and 5 % very long segments). Such regimes are
easy to reason about, but they miss irregular shapes of real data. Offset-defined segments describe any ragged
structure: ragged tensors, groups and clusters, rows of sparse matrices and neighborhoods of graph vertices, among the
uses found in the [usage study](usage_pattern_study.md). Sparse matrices and graphs are well represented in real data
collections and can serve as examples of real irregular length distributions: in their Compressed Sparse Row (CSR) form
[[2]](#ref-2), [[18]](#ref-18), the row offsets are exactly the offsets of this algorithm, and a row sum or a vertex
degree computation is a segmented reduction over them. They have nearly uniform rows (finite element meshes
[[19]](#ref-19)), heavy-tailed rows (web and social graphs [[7]](#ref-7)), and empty rows: isolated vertices, which
have no edges [[16]](#ref-16), and structurally singular matrices, which are singular whatever the values of their
nonzeros, e.g. because a row has no nonzeros at all [[5]](#ref-5). The other uses of segmented reduction are covered
by the synthetic regimes.

For example, the matrix

```
    | 1 0 2 0 |
A = | 0 0 0 0 |
    | 3 4 0 5 |
```

is stored in the CSR form as three arrays: the nonzeros row by row, the column of each of them (counted from 0), and
the offsets at which the rows start. Row 0 has 1 in column 0 and 2 in column 2, row 1 has no nonzeros, and row 2 has 3,
4 and 5 in columns 0, 1 and 3:

```
values      = [1, 2, 3, 4, 5]
columns     = [0, 2, 0, 1, 3]
row_offsets = [0, 2, 2, 5]    // row r is values[row_offsets[r] .. row_offsets[r + 1])
```

The row sums are the segmented reduction of `values` with `row_offsets` as the offsets. The segment lengths are the
row lengths 2, 0 and 3, and the empty row gets the init value:

```cpp
oneapi::dpl::experimental::reduce_by_segment(policy, values, values + 5, sums, 3, row_offsets, 0);
// sums = [1 + 2, 0, 3 + 4 + 5] = [3, 0, 12]
```

A sparse matrix-vector product `y = A * x` reduces the products `values[k] * x[columns[k]]` over the same segments,
and in a graph stored in this form a row is the neighborhood of a vertex. A benchmark needs only the row offsets: they
define the segments, while the values can be generated. The data collections store the matrices as lists of entries
(the Matrix Market coordinate format [[1]](#ref-1)); the conversion described below builds the row offsets from them.

This document reviews how the literature and other libraries generate inputs for segmented reductions and the closely
related sparse matrix-vector multiplication (SpMV) [[8]](#ref-8), [[6]](#ref-6), [[2]](#ref-2), and proposes a
benchmark design for oneDPL that combines parameterized synthetic distributions with a fixed set of real matrices and
graphs.

## Review of Data Generation Approaches

### Uniform lengths with a swept average (ModernGPU)

ModernGPU [[20]](#ref-20) benchmarks its segmented reduction on segments whose lengths are drawn uniformly from `[1, 2 *
avg]` (or `[0, 2 * avg]` when empty segments are allowed), so that the average length is `avg`. One series fixes the
average at 500 and varies the input size; the other fixes the input size at 20M elements and sweeps the average over 10,
20, 50, ..., 500K. The sweep compares implementations over the whole range of lengths rather than at a few points.

### Fixed and bounded-uniform lengths (CUB)

The CUB benchmarks of CCCL [[21]](#ref-21) (`cub/benchmarks/bench/segmented_reduce`) measure
`cub::DeviceSegmentedReduce` with fixed-size segments of `2^0..2^16` elements and with variable-size segments uniformly
distributed in `[1, MaxSegmentSize]` for `MaxSegmentSize = 2^1..2^16`, on inputs of `2^16..2^28` elements. Like
ModernGPU, they cover a wide range of lengths but no skewed distributions and no empty segments.

### Real sparse matrices (SpMV literature)

Williams et al. [[6]](#ref-6) selected 14 matrices with diverse row structures for SpMV on multicore platforms; Bell and
Garland [[8]](#ref-8) adopted the same set for GPUs, and ModernGPU [[20]](#ref-20) uses it in its CSR SpMV benchmark.
All of them (except a generated dense 2000 x 2000 matrix) come from the University of Florida, now SuiteSparse, Matrix
Collection [[11]](#ref-11). In the terms of this algorithm, a row is a segment: "Rows" is the number of segments,
"Nonzeros" is the input size, and "Row length" is the segment length, whose average is Nonzeros / Rows.

| Matrix | Domain | Rows | Nonzeros | Row length: avg / max |
|---|---|---|---|---|
| dense2 | dense (generated) | 2,000 | 4,000,000 | 2,000 / 2,000 |
| pdb1HYS | protein | 36,417 | 4,344,765 | 119 / 204 |
| consph | FEM, spheres | 83,334 | 6,010,480 | 72 / 81 |
| cant | FEM, cantilever | 62,451 | 4,007,383 | 64 / 78 |
| pwtk | wind tunnel | 217,918 | 11,634,424 | 53 / 180 |
| rma10 | FEM, harbor | 46,835 | 2,374,001 | 51 / 145 |
| conf5_4-8x8-05 | QCD | 49,152 | 1,916,928 | 39 / 39 |
| shipsec1 | FEM, ship | 140,874 | 7,813,404 | 55 / 102 |
| mac_econ_fwd500 | economics | 206,500 | 1,273,389 | 6 / 44 |
| mc2depi | epidemiology | 525,825 | 2,100,225 | 4 / 4 |
| cop20k_A | FEM, accelerator | 121,192 | 2,624,331 | 22 / 81, 21,349 empty rows |
| scircuit | circuit | 170,998 | 958,936 | 6 / 353 |
| webbase-1M | web graph | 1,000,005 | 3,105,536 | 3 / 4,700 |
| rail4284 | linear programming | 4,284 | 11,284,032 | 2,634 / 56,182 |

FEM stands for the finite element method [[19]](#ref-19), QCD for quantum chromodynamics [[22]](#ref-22).

The counts are those of the CSR form, with symmetric matrices expanded to both triangles. Merrill and Garland
[[14]](#ref-14) evaluate the merge-based SpMV of CUB on the whole collection, several thousand matrices, which removes
the selection bias of a small set at the cost of benchmark time.

For this proposal, the set matters as a collection of length distributions rather than of matrices. It has nearly
uniform lengths, with the maximum close to the average (`dense2`, `conf5_4-8x8-05`, `mc2depi`, `consph`, `cant`), a
moderate spread (`pdb1HYS`, `shipsec1`, `rma10`, `pwtk`), a wide spread of short or very long rows (`mac_econ_fwd500`,
`rail4284`), heavy tails, with the maximum far above the average (`scircuit`, `webbase-1M`), and a long run of empty
segments (`cop20k_A`).

### Real graphs (SNAP)

The Stanford Large Network Dataset Collection [[23]](#ref-23) provides social, web, communication and road networks
whose degree distributions [[16]](#ref-16) follow power laws [[7]](#ref-7): most rows are short, some are empty, and a
few rows have hundreds to tens of thousands of entries. Several SNAP graphs are also part of SuiteSparse (group `SNAP`),
for example:

| Graph | Rows | Nonzeros | Row length: avg / max | Empty rows |
|---|---|---|---|---|
| web-Google | 916,428 | 5,105,039 | 6 / 456 | 176,974 |
| soc-LiveJournal1 | 4,847,571 | 68,993,773 | 14 / 20,293 | 539,119 |

### Real workloads in the CUB benchmarks

In 2026 the CUB team studied the same question for `DeviceSegmentedReduce` and `DeviceSegmentedSort` [[24]](#ref-24),
[[25]](#ref-25). They converted SuiteSparse matrices to stored segment offsets, keeping the order of the segments and
the original values, analyzed 261 matrices [[26]](#ref-26), and selected a small set with the team of cuSPARSE
[[27]](#ref-27), together with a few customer workloads [[28]](#ref-28). They characterized the segment sizes
[[29]](#ref-29), including the share of empty and single-entry rows, the Gini coefficient [[30]](#ref-30) and the
entropy [[31]](#ref-31), fitted parametric distributions (those of C++ `<random>`) to them, and compared the performance
on the generated and the real inputs. The conclusion was that parametric distributions are too weak to represent real
data, especially for segmented sort, and that for segmented reduction they work only with a high tolerance; histograms
did not save much space either, so they store the actual segment sizes [[25]](#ref-25), [[32]](#ref-32). Their data,
harness and results are internal; only the plans and the conclusions are public.

### Synthetic power-law graphs (R-MAT, Graph500)

R-MAT [[3]](#ref-3) and the Kronecker generator of Graph500 [[10]](#ref-10) produce graphs with power-law degree
distributions from a few parameters, at any scale and without external data. A simpler alternative for segment lengths
alone is to draw them from a Zipf or Pareto distribution [[7]](#ref-7) with an exponent and a cap. Both reproduce the
skew of real graphs, but not their exact structure.

### Summary

The table below compares the approaches reviewed above: whether they reproduce skewed distributions and empty
segments, whether they need external data, and who uses them.

| Approach | Covers skew | Empty segments | External data | Used by |
|---|---|---|---|---|
| Uniform, swept average | no | optional | no | ModernGPU |
| Fixed / bounded uniform | no | no | no | CUB benchmarks |
| Real matrices (Williams / Bell–Garland set) | partly | one matrix | yes, ~240 MB of archives | SpMV literature, ModernGPU |
| Real graphs (SNAP) | yes | yes | yes, from tens to hundreds of MB | graph analytics literature |
| Real workloads (CUB) | yes | yes | yes, not public | CUB benchmarks (planned) |
| Power-law generators | yes | yes | no | Graph500 |

### Data sources and their users

The subsections above take their real inputs from two collections only: SuiteSparse for the matrices and SNAP for the
graphs. Sparse and graph libraries use a few more collections for their benchmarks, and this section lists them, so
that the choice made here can be compared with common practice and the set can be extended later (see
[Open Questions](#open-questions)).

For each collection, the table gives where to get it, what it contains, and who uses it, with references to the
publications or the code. The last column shows which inputs are common in the literature. Using the same inputs lets us
relate our results to what is known about them, e.g. which of them are hard to balance, and run other libraries, such as
CUB or ModernGPU, on exactly the same segments. The published numbers themselves are not directly comparable: they are
mostly for SpMV or graph algorithms, on other hardware. Many of the graphs of these collections are also available in
SuiteSparse in the Matrix Market format, and SNAP provides plain edge lists, so the conversion described in the
[Proposal](#proposal) can read them. The Graph500 generator is different: it produces graphs instead of storing them.

| Source | Link | Content | Used, for example, by |
|---|---|---|---|
| SuiteSparse Matrix Collection [[11]](#ref-11) | https://sparse.tamu.edu (files also on the mirror https://www.cise.ufl.edu/research/sparse/) | about 2,900 sparse matrices from science and engineering, plus graphs mirrored from other collections | Williams et al. [[6]](#ref-6), Bell and Garland [[8]](#ref-8), ModernGPU [[20]](#ref-20), Merrill and Garland (CUB SpMV) [[14]](#ref-14), CSR5 [[12]](#ref-12), the benchmarks of Ginkgo through its `ssget` tool [[17]](#ref-17) |
| SNAP [[23]](#ref-23) | https://snap.stanford.edu/data | social, web, citation, communication, road and collaboration networks | graph analytics literature, Gunrock [[15]](#ref-15); many SNAP graphs are mirrored in SuiteSparse (group `SNAP`) |
| Laboratory for Web Algorithmics (LAW) [[4]](#ref-4) | https://law.di.unimi.it/datasets.php | large web and social crawls (e.g. `uk-2002`, `sk-2005`) | GAP Benchmark Suite (`sk-2005`) [[13]](#ref-13); several LAW graphs are mirrored in SuiteSparse (group `LAW`) |
| DIMACS Implementation Challenges, 9th and 10th [[9]](#ref-9) | https://sparse.tamu.edu/DIMACS10 (the 10th challenge graphs in SuiteSparse, e.g. `road_usa`; original site https://sites.cc.gatech.edu/dimacs10/) | road networks, meshes, clustering and partitioning instances | GAP Benchmark Suite (the road network of the USA from the 9th challenge) [[13]](#ref-13) |
| Graph500 Kronecker generator [[10]](#ref-10) | https://graph500.org | synthetic power-law graphs of any scale, R-MAT style [[3]](#ref-3) | the Graph500 ranking, GAP Benchmark Suite (`kron`) [[13]](#ref-13) |
| Matrix Market [[1]](#ref-1) | https://math.nist.gov/MatrixMarket | an older matrix collection and the coordinate file format used by SuiteSparse | the input format of most sparse libraries and of this proposal's conversion |

### Statistics of the real data sets

To choose the parameters of the synthetic regimes, this section describes the segment lengths of the sixteen real
inputs: the matrices and the two SNAP graphs above. All values are computed from the row lengths that the benchmarks
read. The first table gives the distribution of the lengths over the segments, the second one the share of the
elements by segment length, and the observations after them set the ranges of the [input regimes](#input-regimes) of
the proposal.

The columns of the first table:

- The percentiles [[33]](#ref-33) are over the segments.
- cv is the coefficient of variation [[34]](#ref-34) of the length: the standard deviation divided by the average.
- Gini is the Gini coefficient [[30]](#ref-30) of the lengths: 0 when all segments have the same length, close to 1
  when a few segments hold most of the elements.
- "Length 1" is the share of the segments of one element.
- "Longest empty run" is the largest number of consecutive empty segments.

The rows are sorted by cv.

| Input | Elements | Segments | Length: avg | p10 / p50 / p90 / p99 | max | cv | Gini | Length 1 | Empty segments | Longest empty run |
|---|---|---|---|---|---|---|---|---|---|---|
| dense2 | 4,000,000 | 2,000 | 2,000 | 2,000 / 2,000 / 2,000 / 2,000 | 2,000 | 0.00 | 0.00 | 0 | 0 | 0 |
| conf5_4-8x8-05 | 1,916,928 | 49,152 | 39.0 | 39 / 39 / 39 / 39 | 39 | 0.00 | 0.00 | 0 | 0 | 0 |
| mc2depi | 2,100,225 | 525,825 | 4.0 | 4 / 4 / 4 / 4 | 4 | 0.02 | 0.00 | 0 | 0 | 0 |
| pwtk | 11,634,424 | 217,918 | 53.4 | 54 / 54 / 54 / 66 | 180 | 0.09 | 0.02 | 0 | 0 | 0 |
| shipsec1 | 7,813,404 | 140,874 | 55.5 | 36 / 54 / 72 / 90 | 102 | 0.20 | 0.09 | 0 | 0 | 0 |
| cant | 4,007,383 | 62,451 | 64.2 | 49 / 74 / 76 / 76 | 78 | 0.22 | 0.11 | 0.4 % | 0 | 0 |
| consph | 6,010,480 | 83,334 | 72.1 | 45 / 81 / 81 / 81 | 81 | 0.26 | 0.10 | 4.4 % | 0 | 0 |
| pdb1HYS | 4,344,765 | 36,417 | 119 | 75 / 123 / 156 / 177 | 204 | 0.27 | 0.15 | 0 | 0 | 0 |
| rma10 | 2,374,001 | 46,835 | 50.7 | 22 / 46 / 100 / 115 | 145 | 0.55 | 0.30 | 0 | 0 | 0 |
| cop20k_A | 2,624,331 | 121,192 | 21.7 | 0 / 23 / 45 / 55 | 81 | 0.64 | 0.33 | 0 | 21,349 (17.6 %) | 21,349 |
| mac_econ_fwd500 | 1,273,389 | 206,500 | 6.2 | 3 / 5 / 10 / 24 | 44 | 0.72 | 0.33 | <0.1 % | 0 | 0 |
| scircuit | 958,936 | 170,998 | 5.6 | 3 / 5 / 8 / 22 | 353 | 0.78 | 0.25 | <0.1 % | 0 | 0 |
| web-Google | 5,105,039 | 916,428 | 5.6 | 0 / 3 / 14 / 23 | 456 | 1.18 | 0.57 | 14.0 % | 176,974 (19.3 %) | 9 |
| rail4284 | 11,284,032 | 4,284 | 2,634 | 88 / 1,159 / 6,987 / 19,700 | 56,182 | 1.60 | 0.65 | 0 | 0 | 0 |
| soc-LiveJournal1 | 68,993,773 | 4,847,571 | 14.2 | 0 / 4 / 36 / 142 | 20,293 | 2.54 | 0.72 | 18.9 % | 539,119 (11.1 %) | 84 |
| webbase-1M | 3,105,536 | 1,000,005 | 3.1 | 2 / 2 / 3 / 17 | 4,700 | 8.16 | 0.35 | 0.6 % | 0 | 0 |

The time of a segmented reduction depends on both its elements and its segments [[14]](#ref-14), [[20]](#ref-20). Every
element is read and combined, and every segment adds work of its own: at least reading its bounds and writing its
result, and often more, depending on how the implementation handles segments of different lengths. On top of that, every
call has a fixed cost. How much each part weighs differs between implementations and inputs, so the statistics describe
both the segments and the elements: the first table describes the segments, and the next one shows how the elements are
spread over the segments of different lengths, as the share of the elements, in percent, that lie in the segments of
each length range.

| Input | 1-3 | 4-15 | 16-63 | 64-255 | 256-1K | 1K-4K | 4K-16K | 16K-64K |
|---|---|---|---|---|---|---|---|---|
| dense2 |  |  |  |  |  | 100 |  |  |
| conf5_4-8x8-05 |  |  | 100 |  |  |  |  |  |
| mc2depi | <1 | 100 |  |  |  |  |  |  |
| pwtk | <1 | <1 | 98 | 2 |  |  |  |  |
| shipsec1 |  |  | 75 | 25 |  |  |  |  |
| cant | <1 |  | 30 | 70 |  |  |  |  |
| consph | <1 |  | 9 | 91 |  |  |  |  |
| pdb1HYS |  |  | 3 | 97 |  |  |  |  |
| rma10 |  | <1 | 53 | 47 |  |  |  |  |
| cop20k_A |  | 2 | 96 | 1 |  |  |  |  |
| mac_econ_fwd500 | 13 | 69 | 17 |  |  |  |  |  |
| scircuit | 5 | 85 | 9 | 1 | <1 |  |  |  |
| web-Google | 10 | 61 | 28 | 1 | <1 |  |  |  |
| rail4284 | <1 | <1 | <1 | 1 | 6 | 28 | 49 | 16 |
| soc-LiveJournal1 | 4 | 17 | 41 | 30 | 8 | <1 | <1 | <1 |
| webbase-1M | 61 | 13 | 8 | 9 | 3 | 5 | <1 |  |

Observations:

- **Averages.** The average lengths range from 3 to about 2,600; five of the sixteen inputs have an average below 10
  (`webbase-1M`, `mc2depi`, `scircuit`, `web-Google`, `mac_econ_fwd500`).
- **Input size and length.** The input sizes span a factor of about 70 (from about 1 to 69 million elements) and the
  averages a factor of about 850; the rank correlation [[35]](#ref-35) between them is weak (0.45 over the sixteen
  inputs). The length is set by the structure of the data rather than by its size, which supports sweeping the average
  at a fixed input size, and the input size separately.
- **Uniformity.** Half of the inputs have much narrower lengths than the uniform distribution `[1, 2 * avg]`, whose cv
  is about 0.58: eight inputs have a cv of at most 0.27, and three of them have practically a single length.
- **Empty segments.** Thirteen inputs have none. The graphs have 11 to 19 % of them, scattered (the longest run is 84
  segments), while in `cop20k_A` all of its 17.6 % form one run.
- **Heavy tails.** Four inputs have a maximum of 60 to 1,500 times the average (`scircuit`, `web-Google`,
  `soc-LiveJournal1`, `webbase-1M`). A maximum likelihood fit of a power law [[7]](#ref-7)
  to the lengths of at least 10 gives exponents from about 2 (`webbase-1M`, `soc-LiveJournal1`) to 3.4 (`web-Google`);
  this is a rough estimate, without a test of how well a power law fits. Even where most segments are short, a
  noticeable share of the elements can lie in long ones: in `webbase-1M`, 61 % of the elements are in segments of 1 to 3
  elements, but 5 % are in segments of 1K to 4K.
- **Frequency of use.** The collections tell which shapes exist, not how often an input size or a shape occurs in
  applications: the Williams / Bell–Garland matrices were selected for diverse row structures, and the usage study lists
  the kinds of applications but not their sizes. The statistics can therefore set the ranges of the regimes, but not
  weights by frequency of use.
- **Inequality.** The Gini coefficient complements the ratio of the maximum to the average: `webbase-1M` has an extreme
  maximum but a moderate Gini (0.35), because its few long rows hold a small share of the elements, while in
  `soc-LiveJournal1` (0.72) the elements are unevenly spread over the whole set. Neither the ratio nor the Gini
  coefficient depends on the order of the segments, so the longest empty run is listed separately.
- **Synthetic regimes and real inputs.** The statistics set the ranges of the synthetic regimes, which cover the shapes
  cheaply in every run, but do not make the real inputs unnecessary: the CUB study described above found that inputs
  generated from fitted distributions reproduce the performance on the real ones only with a high tolerance
  [[25]](#ref-25), [[32]](#ref-32). Hence the real inputs remain part of the suite.

### How the libraries measure segmented reduction

The libraries above measure their segmented reductions with different benchmark harnesses. The harnesses differ in what
they time, in the state of the caches, and in when they stop, so the numbers of two libraries are comparable only where
these choices agree. This section describes the harnesses whose code is open; the table at its end compares them.

#### CUB and NVBench

The benchmarks of `cub::DeviceSegmentedReduce` [[36]](#ref-36) run on NVBench [[37]](#ref-37), the benchmarking library
of NVIDIA for CUDA kernels. The variable-size benchmark sweeps the input size over `2^16`, `2^20`, `2^24` and `2^28`
elements and the maximum segment size over `2^1..2^16`, with the lengths drawn uniformly from `[1, MaxSegmentSize]`, for
several value and offset types; other benchmarks cover fixed-size segments, a guaranteed maximum segment size, and
argmin and argmax. The memory traffic counted for the bandwidth includes the values, the offsets and the results.

NVBench takes cold measurements by default:

- Before each sample, the L2 cache of the device is flushed, and each sample runs the benchmark once.
- The GPU time of a sample is taken with CUDA events, while a blocking kernel holds the stream until the events and the
  launch are queued, so that the time does not include the delay before the GPU starts.
- The samples are collected until their relative standard deviation falls below 0.5 % or stabilizes, after at least 10
  samples and 0.5 seconds of measured time, or until a timeout of 15 seconds; the GPU clock is watched, and throttled
  samples are discarded.
- Batch measurements, which run the benchmark back to back and report the average time, are disabled for the segmented
  reduction (`exec_tag::no_batch`).

#### rocPRIM, primbench and Google Benchmark

rocPRIM, the AMD counterpart of CUB, provides `rocprim::segmented_reduce`, which assigns one block of threads to each
segment [[38]](#ref-38). Its benchmark uses synthetic segments only [[39]](#ref-39). The input has a fixed size of 128
MiB, the number of segments is 1, 10, 100, 1,000 or 10,000, and the lengths of the segments are drawn uniformly from
`[0, 2 * avg]`, where `avg` is the input size divided by the number of segments. With 4-byte values, the average length
is then at least about 3,300 elements, so the benchmark has no short segments, no skewed distributions and no real data.
The reported throughput counts only the bytes of the input values, not those of the offsets and the results.

In the ROCm/rocm-libraries repository, where rocPRIM is developed now, the benchmark runs on the primbench library
[[39]](#ref-39), [[40]](#ref-40), which controls the measurement closely:

- The cache of the GPU is cleared before each batch of calls by default, i.e. every batch is a cold run; an option keeps
  the cache warm.
- The calls are timed in batches: the number of calls in a batch is doubled until the batch takes at least a minimum
  time on the GPU, 100 ms for this benchmark.
- The measurement stops when the coefficient of variation [[34]](#ref-34) of the last 10 batch times is below 1 %, or
  after a timeout for a noisy run.
- The temperature of the GPU is kept in a fixed range, by default 50 to 60 °C, by warming or cooling the GPU before the
  batches.
- A small spinning kernel blocks the stream until the start event, the calls and the stop event are all queued, so that
  the measured time does not include the delay before the GPU actually starts the calls.
- One unmeasured call before the first batch warms up the kernel.

Before rocPRIM moved into ROCm/rocm-libraries, its repository ROCm/rocPRIM, now marked as deprecated, measured the same
inputs differently [[41]](#ref-41): with Google Benchmark [[42]](#ref-42), after 5 warm-up calls, each measured
iteration timed a fixed batch of 10 calls with a pair of HIP events, without cache clearing, temperature control or a
noise-based stop.

#### ModernGPU

ModernGPU 1.1 [[43]](#ref-43) benchmarks its segmented reduction with its own timer: CUDA events recorded around a loop
of repeated calls, with a device synchronization at the start and at the end, e.g. 500 calls on an input of 20 million
elements. The lengths are drawn uniformly up to twice the average, and the throughput is reported in elements per
second. The caches are not flushed, so the repeated calls run warm, and the loop hides most of the fixed cost of a call.

#### oneDPL

The oneDPL benchmarks time every call on the host with a steady clock, and every call waits for its completion, so the
fixed cost of a call is part of each sample. After warm-up runs, the number of trials is chosen to fill a target time,
and the median is reported. A build option selects the state of the caches: warm, flushed before each trial, or with the
input regenerated for each trial.

#### Comparison

| Harness | Timer | Calls per sample | Caches | Stop | Traffic counted |
|---|---|---|---|---|---|
| NVBench (CUB) | CUDA events, blocking kernel | 1 | cold (L2 flushed) | relative standard deviation below 0.5 %, or a timeout | values, offsets, results |
| primbench (rocPRIM now) | HIP events, blocking kernel | a batch, doubled up to a minimum time | cold by default | coefficient of variation below 1 %, or a timeout | values |
| Google Benchmark (rocPRIM before the move) | HIP events | a batch of 10 | warm | iterations chosen by Google Benchmark | values |
| ModernGPU 1.1 | CUDA events | a loop of hundreds of calls | warm | a fixed number of calls | elements |
| oneDPL | host steady clock | 1, synchronous | warm, flushed or regenerated | a target time after warm-up | time only |

Two differences matter most for a comparison with oneDPL. A batch or a loop of calls queued in one stream hides most of
the fixed cost of a call, while oneDPL and the cold NVBench samples include it. And a cold cache gives longer times than
a warm one for inputs that fit into the cache. The numbers of different harnesses are therefore comparable only on large
inputs, unless the same harness runs both libraries, as it does for the CUB counterpart of the oneDPL benchmarks (see
[File format of the segments](#file-format-of-the-segments)).

## Proposal

### Input regimes

The benchmark suite of `reduce_by_segment` with offsets keeps its current synthetic regimes and adds:

1. **An average length sweep** in the ModernGPU style: lengths uniform in `[1, 2 * avg]` for
   `avg = 2, 5, 10, 20, 50, ..., 500K` at a fixed input size (`2^24` elements); it starts below 10 because five of the
   real inputs have such averages. It reuses the existing generator, which already takes the bounds of the lengths, and
   only adds a list of regimes to the runner.
2. **Narrow lengths**: the same averages with lengths uniform in `[0.75 * avg, 1.25 * avg]` (cv about 0.14) and with a
   fixed length, like the half of the real inputs that are much narrower than the sweep above.
3. **Empty segments**: the sweep with a given share of empty segments, 10 and 20 %, placed at random, as in the real
   graphs (11 to 19 %); lengths in `[0, 2 * avg]` alone would give only about `1 / (2 * avg + 1)` of them. A long run of
   empty segments, as in `cop20k_A`, is a regime of its own: one run of `R` empty segments among short ones. The
   generator must allow more segments than elements; it currently assumes that the number of segments is at most the
   input size.
4. **A power-law regime**: Zipf-distributed lengths [[7]](#ref-7) with the exponents 2, 2.5, 3 and 3.5 (the fit to the
   real inputs gives about 2 to 3.4) and caps such that the maximum is 60 to 1,500 times the average, as in the real
   inputs with heavy tails, seeded deterministically.
5. **Real segments from files**: the CSR row offsets of the 13 matrices of the Williams / Bell–Garland set, the
   generated `dense2`, and SNAP graphs (`web-Google`, `soc-LiveJournal1`, optionally `com-Orkut`).

The values are generated by the benchmark in all regimes; only the structure of the segments comes from a file.

### File format of the segments

A segments file is little-endian binary: the 8-byte magic `ONEDPLSG`, a `uint64` version (1), a `uint64` number of
segments `m`, a `uint64` number of elements `nnz`, and `m + 1` `int64` offsets with `offsets[0] = 0` and
`offsets[m] = nnz`. A conversion script produces it from the Matrix Market files of SuiteSparse and from the edge lists
of SNAP, expanding symmetric matrices to both triangles and keeping empty rows. A manifest records the source URL, the
checksum of the source archive and of the result, the application and the reason the input is in the set, and the
statistics of the rows: `m`, `nnz`, the number of empty rows, and the minimum, average, median, 99th percentile and
maximum row length. The whole set above takes about 65 MB.

The benchmark selects a file through an environment variable (e.g. `ONEDPLBENCH_SEGMENTS_FILE`); the input size is
then taken from the file. The same mechanism serves the key-based `reduce_by_segment`, whose keys are derived from the
offsets, and the CUB counterpart, so that all three run on identical segments (the key-based version only on the
inputs without empty rows, see below).

### Preparation of the data, step by step

The data set can be prepared once, by a script, from the following sources. The base URL of the SuiteSparse archives is
`https://www.cise.ufl.edu/research/sparse/MM/` (the University of Florida mirror of the collection); behind a proxy,
`http_proxy` and `https_proxy` must be set.

| # | Archive (`<group>/<name>.tar.gz` under the base URL) | Size, MB | Matrix file in the archive |
|---|---|---|---|
| 1 | `Williams/pdb1HYS` | 18 | `pdb1HYS/pdb1HYS.mtx` |
| 2 | `Williams/consph` | 32 | `consph/consph.mtx` |
| 3 | `Williams/cant` | 7 | `cant/cant.mtx` |
| 4 | `Boeing/pwtk` | 56 | `pwtk/pwtk.mtx` |
| 5 | `Bova/rma10` | 18 | `rma10/rma10.mtx` |
| 6 | `QCD/conf5_4-8x8-05` | 15 | `conf5_4-8x8-05/conf5_4-8x8-05.mtx` |
| 7 | `DNVS/shipsec1` | 21 | `shipsec1/shipsec1.mtx` |
| 8 | `Williams/mac_econ_fwd500` | 11 | `mac_econ_fwd500/mac_econ_fwd500.mtx` |
| 9 | `Williams/mc2depi` | 10 | `mc2depi/mc2depi.mtx` |
| 10 | `Williams/cop20k_A` | 9 | `cop20k_A/cop20k_A.mtx` |
| 11 | `Hamm/scircuit` | 8 | `scircuit/scircuit.mtx` |
| 12 | `Williams/webbase-1M` | 10 | `webbase-1M/webbase-1M.mtx` |
| 13 | `Mittelmann/rail4284` | 27 | `rail4284/rail4284.mtx` |
| 14 | `SNAP/web-Google` | 22 | `web-Google/web-Google.mtx` |
| 15 | `SNAP/soc-LiveJournal1` | 243 | `soc-LiveJournal1/soc-LiveJournal1.mtx` |

1. Download every archive of the table with an HTTPS GET of `<base URL><group>/<name>.tar.gz` into a local cache
   directory; an archive already in the cache is not downloaded again.
2. Verify the checksum of every archive against the one recorded with the data, and stop on a mismatch: a changed
   source must be reviewed and its checksum updated explicitly.
3. Extract only the matrix file `<name>/<name>.mtx` from the archive; the other files of an archive (right-hand sides,
   coordinates, node names) are not used.
4. Read the Matrix Market header `%%MatrixMarket matrix coordinate <field> <symmetry>`, skip the comment lines starting
   with `%`, and read the size line `<rows> <columns> <entries>`.
5. For every entry line `<i> <j> [<value>...]` (1-based indices), increment the length of row `i`; if `<symmetry>` is
   `symmetric`, `skew-symmetric` or `hermitian` and `i != j`, also increment the length of row `j`, so that the
   result describes the full matrix. The values are ignored.
6. Check that the number of entry lines equals `<entries>` of the size line.
7. Compute the `m + 1` offsets, `m = <rows>`: `offsets[0] = 0` and `offsets[r + 1] = offsets[r] + ` the length of row
   `r`, and write the segments file in the format above, rows without entries giving empty segments.
8. Generate `dense2` without a download: 2,000 segments of 2,000 elements.
9. Write the manifest: per data set, the source URL, the checksum of the source archive and of the segments file, and
   the statistics of the rows.

Optionally, a larger SNAP graph not mirrored by the collection, `com-Orkut`, can be added from
`https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz` (447 MB): an undirected edge list
`<u> <v>` with comment lines starting with `#`; every edge then increments the lengths of both rows `u` and `v`.
It is not part of the prototype.

### Delivery of the data

Two options are considered:

- **A separate data repository included as a git submodule.** It holds the converted offsets files, the conversion
  script and the manifest, pinned by the submodule commit, so every run uses the same data. It requires hosting a new
  repository and a review of the licenses of the third-party data (the SuiteSparse Matrix Collection is distributed
  under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and asks for a citation of [[11]](#ref-11); the terms
  of the SNAP datasets must be checked).
- **Download on the benchmark machine with a persistent cache.** The runner downloads the archives once per machine,
  verifies their checksums and converts them. It needs no new repository, but depends on the availability of the
  external servers on the first run of every machine.

Downloading on every run is not proposed: the archives total about 0.5 GB, and an unavailable server would break the
runs. Note that the file links of the SuiteSparse web site currently fail with an access error; the files are
available from the University of Florida mirror (`https://www.cise.ufl.edu/research/sparse/MM/<group>/<name>.tar.gz`).

### Metrics and reporting

For every regime and device the benchmark reports the median time, the effective bandwidth
`(nnz * sizeof(value) + (m + 1) * sizeof(offset) + m * sizeof(result)) / time`, and the ratio to the key-based
`reduce_by_segment` on the same segments. Comparisons with other libraries on other hardware, such as CUB, are
described in the next section. A report shows, per device,
the average length sweep as a line chart, and the real data as charts of the ratio and of the bandwidth, sorted by the
average row length, with the tables of all values.

The key-based `reduce_by_segment` cannot express empty segments, so on inputs with empty rows it produces fewer results
and is excluded from the comparison there.

### What the time of a call consists of

The time of one call of a device algorithm consists of several stages:

1. **Compilation of the kernels.** Unless the kernels are compiled ahead of time, they are compiled when they are used
   for the first time.
2. **Preparation on the host.** The implementation queries the properties of the device and chooses the kernels and
   their geometry from the input size and the number of segments.
3. **Allocation of temporary memory.** Some kernels need memory for partial results, allocated and freed in every call.
4. **Launches of the kernels.** Each kernel has a fixed cost to start. Depending on the input, an implementation runs
   one kernel or several: for example, one kernel for the segments, plus one more to combine the partial results of the
   work-groups, or separate kernels for short and long segments.
5. **Transfers between the host and the device**, when the input or the results are in the memory of the host.
6. **The work on the data.** The kernels read the elements and the offsets, compute and write the results; the reading
   and the computations overlap.
7. **Reads back to the host during the call.** An implementation may read a small value back, e.g. the maximum length of
   a segment to choose how to split the long ones, and wait for it.
8. **Waiting for the completion.** The call returns when the results are ready, so its time includes the wait.

The benchmarks measure the stages that belong to the algorithm: the launches of the kernels, the reading of the data,
the work on it and the writing of the results, together with the preparation on the host, the allocations, the reads
back to the host and the wait that the implementation needs for them. The compilation (1) and the transfers between the
host and the device (5) are not measured: the benchmarks exclude the compilation by warm-up runs and keep the input in
the memory of the device. The compilation is paid once per application, and its cost depends on the compiler and on the
caching of the compiled kernels rather than on the algorithm; the transfers depend on where the application keeps its
data, and any implementation of the algorithm would pay the same for them.

The memory bandwidth of the compared GPUs differs, and on large inputs the reading of the data and the writing of the
results take most of the time. To compare the efficiency of the algorithm itself, this transfer between the memory of
the device and its compute units has to be separated from the rest and normalized. It is measured as the time `R(B)` of
an ideal call that only reads the bytes of the call (below); such a call also contains one kernel launch and the wait,
so the normalization replaces them too. Everything else, i.e. the preparation on the host, the allocations, the launches
beyond the first one, the computations, the memory traffic beyond the ideal reading, and the reads back to the host, is
the work of the algorithm and its implementation and is compared as measured.

The normalization replaces only the ideal call. Everything beyond it, e.g. each extra kernel launch, an allocation or a
read back to the host, is still paid at the prices of the device where it runs, which differ from those of the other
device; so the normalized time is not the time of the same code on the other device, only an estimate of it.

### Normalization of the times across devices

oneDPL and CUB run on different GPUs. When we compare their raw times, we compare both the code and the GPU, and we
cannot tell which of the two causes a difference.

Why a normalization is needed:

- For large inputs, the time is mostly the time to read the data from memory. A GPU with faster memory is faster with
  any good code, so on large inputs the raw ratio mostly shows which GPU has faster memory.
- For small inputs, the time is mostly the fixed cost of a call: starting the kernels and waiting for them. The speed of
  the memory does not matter there.
- To find where our code should be improved, what the GPU itself costs (the speed of its memory and one kernel start)
  has to be removed from the comparison, so that only the extra work of the code remains.

The options:

| Option | How | Problem |
|---|---|---|
| Raw times | compare the times as measured | mixes the code and the GPU |
| Peak bandwidth from the specification | divide by the bandwidth from the data sheet | GPUs reach very different shares of it; wrong for small inputs |
| Measured bandwidth for the whole time | scale the whole time by the ratio of the measured bandwidths | also scales the fixed cost, so it is wrong for small inputs |
| Efficiency | compare `R(B) / T` of each library, each on its GPU | a share, not a time |
| Normalization of the transfer only | replace the read time on our GPU with the read time on the other GPU, keep the fixed cost | the kernel start of each driver stays in the comparison, while the ideal call below removes it |
| Normalization of the ideal call | replace the time of an ideal call on our GPU with that on the other GPU (below) | an estimate, see the notes |

The proposal is to show three numbers for every case: the raw time, the time normalized by the ideal call, and the
efficiency of the call against the ideal one. The normalized time is:

```
T_norm = T - R(B) + R_ref(B)
```

- `T` is the measured time of our call.
- `B` is the number of bytes the call reads and writes, as in the effective bandwidth above.
- `R(B)` is the time of an ideal call on our GPU that only reads `B` bytes: one kernel that does nothing but read,
  measured in the same run and on the same machine as the benchmarks.
- `R_ref(B)` is the same on the other GPU.

In words: from our time we take away what even an ideal call needs on our GPU, and add what an ideal call needs on the
other GPU. What is left of our time is the extra work of our code: more kernels, waits, computations. That is the part
we can improve. For large inputs this compares how close each library gets to the speed of the memory of its GPU; for
small inputs it compares only the costs above one kernel start, which come from the code and not from the driver.

The efficiency is `E = R(B) / T`: 1 means that the call is as fast as reading its bytes once, 0.5 that it takes twice as
long. It is computed for each library on its own GPU and needs no reference device, so it is the simplest number to
compare: for example, an implementation with an efficiency close to 1 on large inputs reads at the full speed of the
memory of its GPU, whatever that speed is. The normalized time gives the same comparison as a time on the reference GPU.

Notes:

- The ideal read is measured with a dedicated kernel, not with a reduction, because a reduction also computes and
  combines partial results.
- The ideal read on each GPU includes its caches: an input that fits into the cache of one GPU but not of the other is
  read faster on the first one (e.g. an input of a few tens of MB on GPUs with caches of 18 and 32 MB), so the
  normalization takes the difference in the cache sizes away too, as a property of the hardware rather than of the code.
- The normalized time is an estimate: on a GPU, reading and computing overlap.
- The raw times are always shown too, because they are what a user gets.

### Prototype

The format, the conversion of all the data sets above, the file mode of the existing `reduce_by_segment` and
`reduce_by_segment_offsets` benchmarks, the data repository as a submodule (one of the two delivery options above) and
the report have been prototyped locally. The prototype runs the existing synthetic regimes and the real inputs on a GPU
and on a CPU device as SYCL devices; the new regimes 2 to 4 are not implemented yet.

## Open Questions

- Where to host the data repository, and the licensing review of the SuiteSparse and SNAP data.
- Which subset runs in continuous integration, given the time budget, and which runs only on demand.
- Whether to extend the set beyond the Williams / Bell–Garland matrices, e.g. to a sample of the whole SuiteSparse
  collection as in [[14]](#ref-14).
- Value types (`float`, `double`, integer types) and offset types (32-bit and 64-bit) to measure.
- Whether the same data sets should also drive the benchmarks of the host policies.
- Whether to take the values of the real inputs from the matrices instead of generating them: for a sum of
  floating-point values it does not change the performance, but for operations whose work depends on the values (e.g.
  the minimum with its position) it may.
- Whether to add inputs from applications that use segmented reductions (see the usage study), as the CUB set includes
  customer workloads [[28]](#ref-28).

## References

#### Books and articles

1. <a id="ref-1"></a>R. F. Boisvert, R. Pozo, K. Remington, R. F. Barrett and J. J. Dongarra. *Matrix Market: A Web
   Resource for Test Matrix Collections.* In: Quality of Numerical Software, Chapman & Hall, 1997.
   https://math.nist.gov/MatrixMarket
2. <a id="ref-2"></a>Y. Saad. *Iterative Methods for Sparse Linear Systems.* 2nd ed., SIAM, 2003, Chapter 3 (storage
   schemes of sparse matrices). https://doi.org/10.1137/1.9780898718003
3. <a id="ref-3"></a>D. Chakrabarti, Y. Zhan and C. Faloutsos. *R-MAT: A Recursive Model for Graph Mining.* Proceedings
   of the SIAM International Conference on Data Mining, 2004. https://doi.org/10.1137/1.9781611972740.43
4. <a id="ref-4"></a>P. Boldi and S. Vigna. *The WebGraph Framework I: Compression Techniques.* Proceedings of WWW'04,
   2004. https://doi.org/10.1145/988672.988752
5. <a id="ref-5"></a>T. A. Davis. *Direct Methods for Sparse Linear Systems.* SIAM, 2006 (structural rank and
   structurally singular matrices). https://doi.org/10.1137/1.9780898718881
6. <a id="ref-6"></a>S. Williams, L. Oliker, R. Vuduc, J. Shalf, K. Yelick and J. Demmel. *Optimization of Sparse
   Matrix-Vector Multiplication on Emerging Multicore Platforms.* Proceedings of SC'07, 2007.
   https://doi.org/10.1145/1362622.1362674
7. <a id="ref-7"></a>A. Clauset, C. R. Shalizi and M. E. J. Newman. *Power-Law Distributions in Empirical Data.* SIAM
   Review 51(4), 2009 (power laws, Zipf and Pareto distributions, heavy tails, the maximum likelihood fit with a lower
   bound). https://doi.org/10.1137/070710111
8. <a id="ref-8"></a>N. Bell and M. Garland. *Implementing Sparse Matrix-Vector Multiplication on Throughput-Oriented
   Processors.* Proceedings of SC'09, 2009. Also: *Efficient Sparse Matrix-Vector Multiplication on CUDA*, NVIDIA
   Technical Report NVR-2008-004, 2008. https://doi.org/10.1145/1654059.1654078
9. <a id="ref-9"></a>C. Demetrescu, A. V. Goldberg and D. S. Johnson (eds.). *The Shortest Path Problem: Ninth DIMACS
   Implementation Challenge.* DIMACS Series in Discrete Mathematics and Theoretical Computer Science 74, AMS, 2009.
   https://doi.org/10.1090/dimacs/074 D. A. Bader, H. Meyerhenke, P. Sanders and D. Wagner (eds.). *Graph Partitioning
   and Graph Clustering: Tenth DIMACS Implementation Challenge.* Contemporary Mathematics 588, AMS, 2013.
   https://doi.org/10.1090/conm/588
10. <a id="ref-10"></a>R. C. Murphy, K. B. Wheeler, B. W. Barrett and J. A. Ang. *Introducing the Graph 500.* Cray User
    Group, 2010. https://graph500.org
11. <a id="ref-11"></a>T. A. Davis and Y. Hu. *The University of Florida Sparse Matrix Collection.* ACM Transactions on
    Mathematical Software 38(1), 2011. https://doi.org/10.1145/2049662.2049663, https://sparse.tamu.edu
12. <a id="ref-12"></a>W. Liu and B. Vinter. *CSR5: An Efficient Storage Format for Cross-Platform Sparse Matrix-Vector
    Multiplication.* Proceedings of ICS'15, 2015. https://doi.org/10.1145/2751205.2751209
13. <a id="ref-13"></a>S. Beamer, K. Asanović and D. Patterson. *The GAP Benchmark Suite.* arXiv:1508.03619, 2015.
    https://arxiv.org/abs/1508.03619, https://github.com/sbeamer/gapbs
14. <a id="ref-14"></a>D. Merrill and M. Garland. *Merge-Based Parallel Sparse Matrix-Vector Multiplication.*
    Proceedings of SC'16, 2016. https://doi.org/10.1109/SC.2016.57, abstract and preprint:
    https://github.com/dumerrill/merge-spmv
15. <a id="ref-15"></a>Y. Wang, A. Davidson, Y. Pan, Y. Wu, A. Riffel and J. D. Owens. *Gunrock: A High-Performance
    Graph Processing Library on the GPU.* Proceedings of PPoPP'16, 2016. https://doi.org/10.1145/2851141.2851145,
    https://github.com/gunrock/gunrock
16. <a id="ref-16"></a>R. Diestel. *Graph Theory.* 5th ed., Springer, 2017, Chapter 1 (the degree of a vertex; a vertex
    of degree 0 is isolated). https://doi.org/10.1007/978-3-662-53622-3. Also: *Degree (graph theory).* Wikipedia.
    https://en.wikipedia.org/wiki/Degree_(graph_theory)
17. <a id="ref-17"></a>H. Anzt et al. *Ginkgo: A Modern Linear Operator Algebra Framework for High Performance
    Computing.* ACM Transactions on Mathematical Software 48(1), 2022. https://doi.org/10.1145/3480935,
    https://github.com/ginkgo-project/ginkgo, `ssget`: https://github.com/ginkgo-project/ssget

#### Libraries, repositories and other resources

18. <a id="ref-18"></a>*Sparse matrix: Compressed sparse row.* Wikipedia.
    https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_%28CSR,_CRS_or_Yale_format%29
19. <a id="ref-19"></a>*Finite element method.* Wikipedia. https://en.wikipedia.org/wiki/Finite_element_method
20. <a id="ref-20"></a>S. Baxter. *ModernGPU.* NVIDIA Research. https://github.com/moderngpu/moderngpu (version 1.1:
    `benchmarksegreduce`, `benchmarkspmvcsr`); segmented reduction: https://moderngpu.github.io/segreduce.html
21. <a id="ref-21"></a>NVIDIA CCCL: CUB device-wide segmented reduction and its benchmarks.
    https://github.com/NVIDIA/cccl (`cub/benchmarks/bench/segmented_reduce`).
22. <a id="ref-22"></a>*Quantum chromodynamics.* Wikipedia. https://en.wikipedia.org/wiki/Quantum_chromodynamics
23. <a id="ref-23"></a>J. Leskovec and A. Krevl. *SNAP Datasets: Stanford Large Network Dataset Collection.* 2014.
    https://snap.stanford.edu/data
24. <a id="ref-24"></a>NVIDIA CCCL issue #11250. *[FEA]: collect segmented reduce benchmarks.* 2026.
    https://github.com/NVIDIA/cccl/issues/11250
25. <a id="ref-25"></a>NVIDIA CCCL issue #11251. *[SPIKE]: Investigate a distribution-based workload model for segmented
    algorithm benchmarks.* 2026. https://github.com/NVIDIA/cccl/issues/11251
26. <a id="ref-26"></a>NVIDIA CCCL issue #11024. *[FEA]: Add SuiteSparse Matrix Market workloads for segmented reduce.*
    2026. https://github.com/NVIDIA/cccl/issues/11024
27. <a id="ref-27"></a>NVIDIA. *cuSPARSE.* https://docs.nvidia.com/cuda/cusparse/
28. <a id="ref-28"></a>NVIDIA CCCL issue #11313. *[FEA]: Collect representative DeviceSegmentedReduce workloads.* 2026.
    https://github.com/NVIDIA/cccl/issues/11313
29. <a id="ref-29"></a>NVIDIA CCCL issue #11315. *[SPIKE]: Characterize real segmented-workload distributions.* 2026.
    https://github.com/NVIDIA/cccl/issues/11315
30. <a id="ref-30"></a>*Gini coefficient.* Wikipedia. https://en.wikipedia.org/wiki/Gini_coefficient
31. <a id="ref-31"></a>*Entropy (information theory).* Wikipedia.
    https://en.wikipedia.org/wiki/Entropy_(information_theory)
32. <a id="ref-32"></a>NVIDIA CCCL issue #11317. *[SPIKE]: Compare real and modeled segmented-algorithm performance.*
    2026. https://github.com/NVIDIA/cccl/issues/11317
33. <a id="ref-33"></a>*Percentile.* Wikipedia. https://en.wikipedia.org/wiki/Percentile
34. <a id="ref-34"></a>*Coefficient of variation.* Wikipedia. https://en.wikipedia.org/wiki/Coefficient_of_variation
35. <a id="ref-35"></a>*Spearman's rank correlation coefficient.* Wikipedia.
    https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
36. <a id="ref-36"></a>NVIDIA. *CUB benchmarks of the device-wide segmented reduction.* NVIDIA/cccl,
    https://github.com/NVIDIA/cccl/tree/c0817a7ed2553d6989bf51e200db1966af90b68f/cub/benchmarks/bench/segmented_reduce
37. <a id="ref-37"></a>NVIDIA. *NVBench: CUDA kernel benchmarking library.* https://github.com/NVIDIA/nvbench,
    measurement options:
    https://github.com/NVIDIA/nvbench/blob/73cdee8578f2c038d3213c3178ec290182a3cf7f/docs/cli_help.md
38. <a id="ref-38"></a>AMD. *rocPRIM: device-wide segmented reduction* (`rocprim::segmented_reduce`).
    ROCm/rocm-libraries,
    https://github.com/ROCm/rocm-libraries/blob/6c3abd43f019f2ffd8642f1847cdcbbf8d10e9f4/projects/rocprim/rocprim/include/rocprim/device/device_segmented_reduce.hpp#L83-L116
39. <a id="ref-39"></a>AMD. *rocPRIM benchmark of the device-wide segmented reduction.* ROCm/rocm-libraries,
    https://github.com/ROCm/rocm-libraries/blob/6c3abd43f019f2ffd8642f1847cdcbbf8d10e9f4/projects/rocprim/benchmark/benchmark_device_segmented_reduce.hpp,
    https://github.com/ROCm/rocm-libraries/blob/6c3abd43f019f2ffd8642f1847cdcbbf8d10e9f4/projects/rocprim/benchmark/benchmark_device_segmented_reduce.cpp
40. <a id="ref-40"></a>AMD. *primbench: a single-header HIP and CUDA benchmarking library.* ROCm/rocm-libraries,
    https://github.com/ROCm/rocm-libraries/blob/6c3abd43f019f2ffd8642f1847cdcbbf8d10e9f4/shared/primbench/README.md
41. <a id="ref-41"></a>AMD. *rocPRIM benchmark of the device-wide segmented reduction before the move to
    ROCm/rocm-libraries.* ROCm/rocPRIM (deprecated),
    https://github.com/ROCm/rocPRIM/blob/14cd5e3c27a4b9ae7d510823a450723a03985ac0/benchmark/benchmark_device_segmented_reduce.parallel.hpp,
    https://github.com/ROCm/rocPRIM/blob/14cd5e3c27a4b9ae7d510823a450723a03985ac0/benchmark/benchmark_device_segmented_reduce.cpp
42. <a id="ref-42"></a>Google. *Benchmark: a microbenchmark support library.* https://github.com/google/benchmark
43. <a id="ref-43"></a>S. Baxter. *ModernGPU 1.1: segmented reduction benchmark and timer.*
    https://github.com/moderngpu/moderngpu/blob/4cddabe1201a325fc630ed4ce718583d35658f8e/benchmarksegreduce/benchmarksegreduce.cu,
    https://github.com/moderngpu/moderngpu/blob/4cddabe1201a325fc630ed4ce718583d35658f8e/src/mgpucontext.cu
