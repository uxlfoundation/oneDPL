# Benchmarking Offset-Based Segmented Reduction on Real and Synthetic Segment Distributions

## Introduction

The [Offset-Based Segmented Reduction](../segmented_reduce/README.md) RFC (#2848) adds a `reduce_by_segment` overload
whose segments are defined by `num_segments + 1` offsets. Its performance depends on the distribution of the segment
lengths much more than on the input size alone: an implementation picks different strategies for short, long and mixed
segments, and the worst cases are skewed distributions, where a few very long segments coexist with millions of
short or empty ones.

The benchmarks used so far generate the segments synthetically: the lengths are drawn uniformly from one or two ranges
(e.g. 1..1024, 1..16, 2K..128K, 64K..256K, and a mixture of 95 % short and 5 % very long segments). Such regimes are
easy to reason about, but they miss the shapes that dominate in practice. The main source of offset-defined segments
is the compressed sparse row (CSR) format: the row offsets of a sparse matrix or of a graph adjacency are exactly the
offsets of this algorithm, and a row sum or a vertex degree computation is a segmented reduction over them. Real
matrices and graphs have nearly uniform rows (finite element meshes), heavy-tailed rows (web and social graphs), and
empty rows (isolated vertices, structurally singular matrices).

This document reviews how the literature and other libraries generate inputs for segmented reductions and the closely
related sparse matrix-vector multiplication (SpMV), and proposes a benchmark design for oneDPL that combines
parameterized synthetic distributions with a fixed set of real matrices and graphs.

## Review of Data Generation Approaches

### Uniform lengths with a swept average (ModernGPU)

ModernGPU [6] benchmarks its segmented reduction on segments whose lengths are drawn uniformly from `[1, 2 * avg]`
(or `[0, 2 * avg]` when empty segments are allowed), so that the average length is `avg`. One series fixes the average
at 500 and varies the input size; the other fixes the input size at 20M elements and sweeps the average over
10, 20, 50, ..., 500K. The sweep is a compact way to locate the length thresholds at which an implementation switches
strategies, and to compare implementations along the whole range of lengths rather than at a few points.

### Fixed and bounded-uniform lengths (CUB)

The CUB benchmarks of CCCL [8] (`cub/benchmarks/bench/segmented_reduce`) measure `cub::DeviceSegmentedReduce` with
fixed-size segments of `2^0..2^16` elements and with variable-size segments uniformly distributed in
`[1, MaxSegmentSize]` for `MaxSegmentSize = 2^1..2^16`, on inputs of `2^16..2^28` elements. Like ModernGPU, they cover
a wide range of lengths but no skewed distributions and no empty segments.

### Real sparse matrices (SpMV literature)

Williams et al. [2] selected 14 matrices with diverse row structures for SpMV on multicore platforms; Bell and
Garland [1] adopted the same set for GPUs, and ModernGPU [6] uses it in its CSR SpMV benchmark.
All of them (except a generated dense 2000 x 2000 matrix) come from the University of Florida, now SuiteSparse,
Matrix Collection [4]:

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

## Proposal

### Input regimes

The benchmark suite of `reduce_by_segment` with offsets keeps its current synthetic regimes and adds:

1. **An average length sweep** in the ModernGPU style: lengths uniform in `[1, 2 * avg]` for
   `avg = 10, 20, 50, ..., 500K` at a fixed input size (`2^24` elements). It reuses the existing generator, which
   already takes the bounds of the lengths, and only adds a list of regimes to the runner.
2. **Empty segments**: the same sweep with lengths in `[0, 2 * avg]`. The generator must bound the number of segments,
   which it currently assumes to be at most the input size.
3. **A power-law regime**: Zipf-distributed lengths with an exponent and a cap, seeded deterministically.
4. **Real segments from files**: the CSR row offsets of the 13 matrices of the Williams / Bell–Garland set, the
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
2. Verify the SHA-256 checksum of every archive against the list below, and stop on a mismatch: a changed source must
   be reviewed and its checksum updated explicitly.
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

SHA-256 checksums of the archives, as downloaded on the date of this proposal:

```
cb8681ba8987fd44214b0c25ca499610b710af979e87d9c3e5b2c87aff59790a  Williams/pdb1HYS.tar.gz
a06af28ebb94274fd5cea46d4965e7feba9f0375fa40c9030a9eab45e1e128fd  Williams/consph.tar.gz
58c63a4a0c446c4b57dd3146b0c4dc1bd8416dc4eee2dfbe7dd45cfb3e7181d7  Williams/cant.tar.gz
46e4f709fae9d70c148e1b01ebf55cc63d4a6d9ecc8c5331d2dc0f936a64eb99  Boeing/pwtk.tar.gz
50db8d278d371531b3dd0638444d47a77f3a3e189663993a857861dbc34c5e3f  Bova/rma10.tar.gz
63e4f4839cd739c53cd753a7456db2822a6dba681b20782417bc0f25dcb34eeb  QCD/conf5_4-8x8-05.tar.gz
d021889affed5429f85b606900f76870d0b1b1aefd92529cc6f43bf9d7ef0eb1  DNVS/shipsec1.tar.gz
0dec2952b2908e3d59e4179289245db7f2c84f9e5e6543e818491deed5978f82  Williams/mac_econ_fwd500.tar.gz
c02fef86efdd4f4322487e7472697a3d30b084ede1021e6d6889b347d3f6b268  Williams/mc2depi.tar.gz
cf055e12e788e72d40d625bd848a7b716608e5c6aa892080497efd5f3454bb9e  Williams/cop20k_A.tar.gz
227d4c98e51c8af49c07f89929c62f2523e115e81b672e7f306185ea92c2996f  Hamm/scircuit.tar.gz
17a0391cdd966350b2b41f32aaf8e6684f3c55f25eb68c4be088f44f728a3ed4  Williams/webbase-1M.tar.gz
084d77a0ef65b86cab1766dc7246b43a281fcc96bc3982e515aa76275caa3707  Mittelmann/rail4284.tar.gz
15fa6d28895f030f03c2da0b8c25ab19fd207614e397706d00003264abe37337  SNAP/web-Google.tar.gz
6518202e02a60f4db9828f3f6ce756191436e61c061c44b66e869b7ba28a647b  SNAP/soc-LiveJournal1.tar.gz
```

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
