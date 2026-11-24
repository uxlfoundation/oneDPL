# Benchmarking Offset-Based Segmented Reduction on Real and Synthetic Segment Distributions

Table of contents:

- [Introduction](#introduction)
- [Review of Data Generation Approaches](#review-of-data-generation-approaches)
  - [Uniform lengths with a swept average (ModernGPU)](#uniform-lengths-with-a-swept-average-moderngpu)
  - [Fixed and bounded-uniform lengths (CUB)](#fixed-and-bounded-uniform-lengths-cub)
  - [Real sparse matrices (SpMV literature)](#real-sparse-matrices-spmv-literature)
  - [Real graphs (SNAP)](#real-graphs-snap)
  - [Synthetic power-law graphs (R-MAT, Graph500)](#synthetic-power-law-graphs-r-mat-graph500)
  - [Data sources and their users](#data-sources-and-their-users)
  - [Summary](#summary)
  - [Statistics of the real data sets](#statistics-of-the-real-data-sets)
- [Proposal](#proposal)
  - [Input regimes](#input-regimes)
  - [File format of the segments](#file-format-of-the-segments)
  - [Preparation of the data, step by step](#preparation-of-the-data-step-by-step)
  - [Delivery of the data](#delivery-of-the-data)
  - [Metrics and reporting](#metrics-and-reporting)
  - [Prototype](#prototype)
- [Open Questions](#open-questions)
- [References](#references)

## Introduction

The [Offset-Based Segmented Reduction](README.md) RFC (#2848) adds a `reduce_by_segment` overload
whose segments are defined by `num_segments + 1` offsets. Its performance depends on the distribution of the segment
lengths much more than on the input size alone: the work per segment ranges from nothing to the whole input, which
makes the work hard to distribute evenly, and the hardest cases are skewed distributions, where a few very long
segments coexist with many short or empty ones.

The benchmarks used so far generate the segments synthetically: the lengths are drawn uniformly from one or two ranges
(e.g. 1..1024, 1..16, 2K..128K, 64K..256K, and a mixture of 95 % short and 5 % very long segments). Such regimes are
easy to reason about, but they miss irregular shapes of real data. Offset-defined segments describe any ragged
structure: ragged tensors, groups and clusters, rows of sparse matrices and neighborhoods of graph vertices, among the
uses found in the [usage study](usage_pattern_study.md). Sparse matrices and graphs are well represented in real data
collections and can serve as examples of real irregular length distributions: in their compressed sparse row (CSR) form,
the row offsets are exactly the offsets of this algorithm, and a row sum or a vertex degree computation is a segmented
reduction over them. They have nearly uniform rows (finite element meshes), heavy-tailed rows (web and social graphs),
and empty rows (isolated vertices, structurally singular matrices). The other sources are represented by the synthetic
regimes.

For example, the matrix

```
    | 1 0 2 0 |
A = | 0 0 0 0 |
    | 3 4 0 5 |
```

is stored in the CSR form as three arrays: the nonzeros row by row, their columns, and the offsets at which the rows
start:

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
(the Matrix Market coordinate format); the conversion described below builds the row offsets from them.

This document reviews how the literature and other libraries generate inputs for segmented reductions and the closely
related sparse matrix-vector multiplication (SpMV), and proposes a benchmark design for oneDPL that combines
parameterized synthetic distributions with a fixed set of real matrices and graphs.

## Review of Data Generation Approaches

### Uniform lengths with a swept average (ModernGPU)

ModernGPU [6] benchmarks its segmented reduction on segments whose lengths are drawn uniformly from `[1, 2 * avg]`
(or `[0, 2 * avg]` when empty segments are allowed), so that the average length is `avg`. One series fixes the average
at 500 and varies the input size; the other fixes the input size at 20M elements and sweeps the average over
10, 20, 50, ..., 500K. The sweep compares implementations over the whole range of lengths rather than at a few
points.

### Fixed and bounded-uniform lengths (CUB)

The CUB benchmarks of CCCL [8] (`cub/benchmarks/bench/segmented_reduce`) measure `cub::DeviceSegmentedReduce` with
fixed-size segments of `2^0..2^16` elements and with variable-size segments uniformly distributed in
`[1, MaxSegmentSize]` for `MaxSegmentSize = 2^1..2^16`, on inputs of `2^16..2^28` elements. Like ModernGPU, they cover
a wide range of lengths but no skewed distributions and no empty segments.

### Real sparse matrices (SpMV literature)

Williams et al. [2] selected 14 matrices with diverse row structures for SpMV on multicore platforms; Bell and
Garland [1] adopted the same set for GPUs, and ModernGPU [6] uses it in its CSR SpMV benchmark.
All of them (except a generated dense 2000 x 2000 matrix) come from the University of Florida, now SuiteSparse,
Matrix Collection [4]. In the terms of this algorithm, a row is a segment: "Rows" is the number of segments, "Nonzeros"
is the input size, and "Row length" is the segment length, whose average is Nonzeros / Rows.

| Matrix | Domain | Rows | Nonzeros | Row length: avg / max |
|---|---|---|---|---|
| dense2 | dense (generated) | 2,000 | 4,000,000 | 2000 / 2000 |
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

The counts are those of the CSR form, with symmetric matrices expanded to both triangles. Merrill and Garland [3]
evaluate the merge-based SpMV of CUB on the whole collection, several thousand matrices, which removes the selection
bias of a small set at the cost of benchmark time.

For this proposal, the set matters as a collection of length distributions rather than of matrices. It has nearly
uniform lengths, with the maximum close to the average (`dense2`, `conf5_4-8x8-05`, `mc2depi`, `consph`, `cant`), a
moderate spread (`pdb1HYS`, `shipsec1`, `rma10`, `pwtk`), heavy tails, with the maximum far above the average
(`mac_econ_fwd500`, `rail4284`, `scircuit`, `webbase-1M`), and a long run of empty segments (`cop20k_A`).

### Real graphs (SNAP)

The Stanford Large Network Dataset Collection [5] provides social, web, communication and road networks whose degree
distributions follow power laws: most rows are tiny or empty, and a few rows have tens of thousands of entries.
Several SNAP graphs are also part of SuiteSparse (group `SNAP`), for example:

| Graph | Rows | Nonzeros | Row length: avg / max | Empty rows |
|---|---|---|---|---|
| web-Google | 916,428 | 5,105,039 | 6 / 456 | 176,974 |
| soc-LiveJournal1 | 4,847,571 | 68,993,773 | 14 / 20,293 | 539,119 |

### Synthetic power-law graphs (R-MAT, Graph500)

R-MAT [7] and the Kronecker generator of Graph500 produce graphs with power-law degree distributions from a few
parameters, at any scale and without external data. A simpler alternative for segment lengths alone is to draw them
from a Zipf or Pareto distribution with an exponent and a cap. Both reproduce the skew of real graphs, but not their
exact structure.

### Data sources and their users

The collections below are the usual sources of real sparse matrices and graphs for benchmarking sparse and graph
algorithms; any of them can feed the segments files of this proposal.

| Source | Link | Content | Used, for example, by |
|---|---|---|---|
| SuiteSparse Matrix Collection [4] | https://sparse.tamu.edu (files also on the mirror https://www.cise.ufl.edu/research/sparse/) | about 2,900 sparse matrices from science and engineering, plus graphs mirrored from other collections | Williams et al. [2], Bell and Garland [1], ModernGPU [6], Merrill and Garland (CUB SpMV) [3], CSR5 [9], the benchmarks of Ginkgo through its `ssget` tool [10] |
| SNAP [5] | https://snap.stanford.edu/data | social, web, citation, communication, road and collaboration networks | graph analytics literature, Gunrock [11]; many SNAP graphs are mirrored in SuiteSparse (group `SNAP`) |
| Laboratory for Web Algorithmics (LAW) [12] | https://law.di.unimi.it/datasets.php | large web and social crawls (e.g. `uk-2002`, `sk-2005`) | GAP Benchmark Suite (`sk-2005`) [13]; several LAW graphs are mirrored in SuiteSparse (group `LAW`) |
| DIMACS Implementation Challenges, 9th and 10th [14] | https://sparse.tamu.edu/DIMACS10 (the 10th challenge graphs in SuiteSparse, e.g. `road_usa`; original site https://sites.cc.gatech.edu/dimacs10/) | road networks, meshes, clustering and partitioning instances | GAP Benchmark Suite (the road network of the USA from the 9th challenge) [13] |
| Graph500 Kronecker generator [15] | https://graph500.org | synthetic power-law graphs of any scale, R-MAT style [7] | the Graph500 ranking, GAP Benchmark Suite (`kron`) [13] |
| Matrix Market [16] | https://math.nist.gov/MatrixMarket | an older matrix collection and the coordinate file format used by SuiteSparse | the input format of most sparse libraries and of this proposal's conversion |

### Summary

| Approach | Covers skew | Empty segments | External data | Comparable with |
|---|---|---|---|---|
| Uniform, swept average | no | optional | no | ModernGPU |
| Fixed / bounded uniform | no | no | no | CUB benchmarks |
| Real matrices (Williams / Bell–Garland set) | partly | one matrix | yes, ~240 MB of archives | SpMV literature, ModernGPU, CUSP |
| Real graphs (SNAP) | yes | yes | yes, from tens to hundreds of MB | graph analytics literature |
| Power-law generators | yes | yes | no | Graph500 |

### Statistics of the real data sets

To choose the parameters of the synthetic regimes, the table below gives the statistics of the segment lengths of the
sixteen real inputs (the matrices and the two SNAP graphs above), computed from the row lengths that the benchmarks
read. The percentiles are over the segments; cv is the coefficient of variation of the length (the standard deviation
divided by the average).

| Input | Elements | Segments | Length: avg | p10 / p50 / p90 / p99 | max | cv | Empty segments | Longest empty run |
|---|---|---|---|---|---|---|---|---|
| dense2 | 4,000,000 | 2,000 | 2,000 | 2,000 / 2,000 / 2,000 / 2,000 | 2,000 | 0.00 | 0 | 0 |
| conf5_4-8x8-05 | 1,916,928 | 49,152 | 39.0 | 39 / 39 / 39 / 39 | 39 | 0.00 | 0 | 0 |
| mc2depi | 2,100,225 | 525,825 | 4.0 | 4 / 4 / 4 / 4 | 4 | 0.02 | 0 | 0 |
| pwtk | 11,634,424 | 217,918 | 53.4 | 54 / 54 / 54 / 66 | 180 | 0.09 | 0 | 0 |
| shipsec1 | 7,813,404 | 140,874 | 55.5 | 36 / 54 / 72 / 90 | 102 | 0.20 | 0 | 0 |
| cant | 4,007,383 | 62,451 | 64.2 | 49 / 74 / 76 / 76 | 78 | 0.22 | 0 | 0 |
| consph | 6,010,480 | 83,334 | 72.1 | 45 / 81 / 81 / 81 | 81 | 0.26 | 0 | 0 |
| pdb1HYS | 4,344,765 | 36,417 | 119 | 75 / 123 / 156 / 177 | 204 | 0.27 | 0 | 0 |
| rma10 | 2,374,001 | 46,835 | 50.7 | 22 / 46 / 100 / 115 | 145 | 0.55 | 0 | 0 |
| cop20k_A | 2,624,331 | 121,192 | 21.7 | 0 / 23 / 45 / 55 | 81 | 0.64 | 21,349 (17.6 %) | 21,349 |
| mac_econ_fwd500 | 1,273,389 | 206,500 | 6.2 | 3 / 5 / 10 / 24 | 44 | 0.72 | 0 | 0 |
| scircuit | 958,936 | 170,998 | 5.6 | 3 / 5 / 8 / 22 | 353 | 0.78 | 0 | 0 |
| web-Google | 5,105,039 | 916,428 | 5.6 | 0 / 3 / 14 / 23 | 456 | 1.18 | 176,974 (19.3 %) | 9 |
| rail4284 | 11,284,032 | 4,284 | 2,634 | 88 / 1,159 / 6,987 / 19,700 | 56,182 | 1.60 | 0 | 0 |
| soc-LiveJournal1 | 68,993,773 | 4,847,571 | 14.2 | 0 / 4 / 36 / 142 | 20,293 | 2.54 | 539,119 (11.1 %) | 84 |
| webbase-1M | 3,105,536 | 1,000,005 | 3.1 | 2 / 2 / 3 / 17 | 4,700 | 8.16 | 0 | 0 |

Where the elements are matters for the work more than where the segments are: the next table gives the share of the
elements, in percent, that lie in the segments of each length range.

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
  averages a factor of about 850; the rank correlation between them is weak (0.45 over the sixteen inputs). The length
  is set by the structure of the data rather than by its size, which supports sweeping the average at a fixed input
  size, and the input size separately.
- **Uniformity.** Half of the inputs are much narrower than the uniform distribution `[1, 2 * avg]`, whose cv is about
  0.58: eight inputs have a cv of at most 0.27, and three of them have practically a single length.
- **Empty segments.** Thirteen inputs have none. The graphs have 11 to 19 % of them, scattered (the longest run is 84
  segments), while in `cop20k_A` all of its 17.6 % form one run.
- **Heavy tails.** Four inputs have a maximum of 60 to 1,500 times the average. A maximum likelihood fit of a power law
  to the lengths of at least 10 gives exponents from about 2 (`webbase-1M`, `soc-LiveJournal1`) to 3.4 (`web-Google`);
  this is a rough estimate, without a test of how well a power law fits. Even where most segments are short, a
  noticeable share of the elements can lie in long ones: in `webbase-1M`, 61 % of the elements are in segments of 1 to 3
  elements, but 5 % are in segments of 1K to 4K.
- **Frequency of use.** The collections tell which shapes exist, not how often an input size or a shape occurs in
  applications: the Williams / Bell–Garland matrices were selected for diverse row structures, and the usage study lists
  the kinds of applications but not their sizes. The statistics can therefore set the ranges of the regimes, but not
  weights by frequency of use.

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
   generator must bound the number of segments, which it currently assumes to be at most the input size.
4. **A power-law regime**: Zipf-distributed lengths with the exponents 2, 2.5 and 3 (the range fitted to the real
   graphs) and caps such that the maximum is 60 to 1,500 times the average, as in the real inputs with heavy tails,
   seeded deterministically.
5. **Real segments from files**: the CSR row offsets of the 13 matrices of the Williams / Bell–Garland set, the
   generated `dense2`, and SNAP graphs (`web-Google`, `soc-LiveJournal1`, optionally `com-Orkut`).

The values are generated by the benchmark in all regimes; only the structure of the segments comes from a file.

### File format of the segments

A segments file is little-endian binary: the 8-byte magic `ONEDPLSG`, a `uint64` version (1), a `uint64` number of
segments `m`, a `uint64` number of elements `nnz`, and `m + 1` `int64` offsets with `offsets[0] = 0` and
`offsets[m] = nnz`. A conversion script produces it from the Matrix Market files of SuiteSparse and from the edge lists
of SNAP, expanding symmetric matrices to both triangles and keeping empty rows. A manifest records the source URL, the
checksum of the source archive and of the result, and the statistics of the rows: `m`, `nnz`, the number of empty rows,
and the minimum, average, median, 99th percentile and maximum row length. The whole set above takes about 65 MB.

The benchmark selects a file through an environment variable (e.g. `ONEDPLBENCH_SEGMENTS_FILE`); the input size is
then taken from the file. The same mechanism serves the key-based `reduce_by_segment`, whose keys are derived from the
offsets, and the CUB counterpart, so that all three run on identical segments.

### Preparation of the data, step by step

The data set is prepared once, by a script, from the following sources. The base URL of the SuiteSparse archives is
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
7. Compute the `m + 1` offsets as the exclusive prefix sum of the row lengths, `m = <rows>`, and write the segments
   file in the format above, rows without entries giving empty segments.
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
  under CC BY 4.0 and asks for a citation of [4]; the terms of the SNAP datasets must be checked).
- **Download on the benchmark machine with a persistent cache.** The runner downloads the archives once per machine,
  verifies their checksums and converts them. It needs no new repository, but depends on the availability of the
  external servers on the first run of every machine.

Downloading on every run is not proposed: the archives total about 0.5 GB, and an unavailable server would break the
runs. Note that the file links of the SuiteSparse web site currently fail with an access error; the files are
available from the University of Florida mirror (`https://www.cise.ufl.edu/research/sparse/MM/<group>/<name>.tar.gz`).

### Metrics and reporting

For every regime and device the benchmark reports the median time, the effective bandwidth
`(nnz * sizeof(value) + (m + 1) * sizeof(offset) + m * sizeof(result)) / time`, and the ratio to the key-based
`reduce_by_segment` on the same segments. Comparisons with other libraries on different hardware, such as CUB, use the
fraction of the measured peak memory bandwidth of each device instead of absolute times. A report shows, per device,
the average length sweep as a line chart, and the real data as charts of the ratio and of the bandwidth, sorted by the
average row length, with the tables of all values.

The key-based `reduce_by_segment` cannot express empty segments, so on inputs with empty rows it produces fewer results
and is excluded from the comparison there.

### Prototype

The format, the conversion of all the data sets above, the file mode of the existing `reduce_by_segment` and
`reduce_by_segment_offsets` benchmarks, the data repository as a submodule and the report have been prototyped
locally. The prototype runs all regimes on a GPU and on a CPU device as SYCL devices.

## Open Questions

- Where to host the data repository, and the licensing review of the SuiteSparse and SNAP data.
- Which subset runs in continuous integration, given the time budget, and which runs only on demand.
- Whether to extend the set beyond the Williams / Bell–Garland matrices, e.g. to a sample of the whole SuiteSparse
  collection as in [3].
- Value types (`float`, `double`, integer types) and offset types (32-bit and 64-bit) to measure.
- Whether the same data sets should also drive the benchmarks of the host policies.

## References

1. N. Bell and M. Garland. *Implementing Sparse Matrix-Vector Multiplication on Throughput-Oriented Processors.*
   Proceedings of SC'09, 2009. Also: *Efficient Sparse Matrix-Vector Multiplication on CUDA*, NVIDIA Technical Report
   NVR-2008-004, 2008.
2. S. Williams, L. Oliker, R. Vuduc, J. Shalf, K. Yelick and J. Demmel. *Optimization of Sparse Matrix-Vector
   Multiplication on Emerging Multicore Platforms.* Proceedings of SC'07, 2007.
3. D. Merrill and M. Garland. *Merge-Based Parallel Sparse Matrix-Vector Multiplication.* Proceedings of SC'16, 2016.
4. T. A. Davis and Y. Hu. *The University of Florida Sparse Matrix Collection.* ACM Transactions on Mathematical
   Software 38(1), 2011. https://sparse.tamu.edu
5. J. Leskovec and A. Krevl. *SNAP Datasets: Stanford Large Network Dataset Collection.* 2014.
   https://snap.stanford.edu/data
6. S. Baxter. *ModernGPU.* NVIDIA Research. https://github.com/moderngpu/moderngpu (version 1.1:
   `benchmarksegreduce`, `benchmarkspmvcsr`).
7. D. Chakrabarti, Y. Zhan and C. Faloutsos. *R-MAT: A Recursive Model for Graph Mining.* Proceedings of the SIAM
   International Conference on Data Mining, 2004.
8. NVIDIA CCCL: CUB device-wide segmented reduction and its benchmarks. https://github.com/NVIDIA/cccl
   (`cub/benchmarks/bench/segmented_reduce`).
9. W. Liu and B. Vinter. *CSR5: An Efficient Storage Format for Cross-Platform Sparse Matrix-Vector Multiplication.*
    Proceedings of ICS'15, 2015.
10. H. Anzt et al. *Ginkgo: A Modern Linear Operator Algebra Framework for High Performance Computing.* ACM Transactions
    on Mathematical Software 48(1), 2022. https://github.com/ginkgo-project/ginkgo, `ssget`:
    https://github.com/ginkgo-project/ssget
11. Y. Wang, A. Davidson, Y. Pan, Y. Wu, A. Riffel and J. D. Owens. *Gunrock: A High-Performance Graph Processing
    Library on the GPU.* Proceedings of PPoPP'16, 2016. https://github.com/gunrock/gunrock
12. P. Boldi and S. Vigna. *The WebGraph Framework I: Compression Techniques.* Proceedings of WWW'04, 2004.
13. S. Beamer, K. Asanović and D. Patterson. *The GAP Benchmark Suite.* arXiv:1508.03619, 2015.
    https://github.com/sbeamer/gapbs
14. C. Demetrescu, A. V. Goldberg and D. S. Johnson (eds.). *The Shortest Path Problem: Ninth DIMACS Implementation
    Challenge.* DIMACS Series in Discrete Mathematics and Theoretical Computer Science 74, AMS, 2009.
    D. A. Bader, H. Meyerhenke, P. Sanders and D. Wagner (eds.). *Graph Partitioning and Graph Clustering: Tenth DIMACS
    Implementation Challenge.* Contemporary Mathematics 588, AMS, 2013.
15. R. C. Murphy, K. B. Wheeler, B. W. Barrett and J. A. Ang. *Introducing the Graph 500.* Cray User Group, 2010.
    https://graph500.org
16. R. F. Boisvert, R. Pozo, K. Remington, R. F. Barrett and J. J. Dongarra. *Matrix Market: A Web Resource for Test
    Matrix Collections.* In: Quality of Numerical Software, Chapman & Hall, 1997. https://math.nist.gov/MatrixMarket
