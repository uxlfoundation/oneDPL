# Algorithm Hints in Backends

Knowing which algorithm is invoking the backend can help tune its performance.

## Motivation

For example, both `equal` and `for_each` use `tbb::parallel_for` under the hood.
These two algorithms have very different workloads, so backend-specific parameters such as the optimal grain size
and partitioning strategy should differ between them.

`for_each`, with its arbitrary user-defined function, has an unpredictable workload, so the backend should choose
only the minimal grain size for it, e.g. to use all available cores with heavy functors or aid load balancing with uneven functors.
`equal`, on the other hand, has a much more predictable workload, so it may benefit from a larger, predefined grain size.

When benchmarking `equal` with 2 cores, it was found that it reaches parity with the serial version at ~16K-32K elements,
which could serve as both a good serial cutoff and a grain size.
Below that, the overhead of parallelism is too high, and the parallel algorithm performs orders of magnitude worse
than the serial version.
Experiments show that it is also possible to eke out ~10%-20% performance improvement
for large sizes by tuning the grain size in oneTBB backend in `find` algorithm
(and likely other 13 similar algorithms using `__parallel_find` and `__parallel_or` patterns).
Almost every algorithm except `for_each` and `transform` has a similarly predictable workload, e.g. `copy`, `move`,
`any_of`, `all_of`, `none_of`, `find`, etc. - up to ~50 algorithms in total.

The same applies to algorithms that use `tbb::parallel_reduce` under the hood, such as `is_sorted`, `minmax_element`,
and `reduce`. There are about 12 of them.

Algorithms that use `__parallel_strict_scan` pattern, such as `inclusive_scan`, `copy_if`,
`partition_copy`, `remove_if`, `remove_copy`, `unique_copy`, `set_union` etc.,
can benefit from different grain sizes (tile sizes in this case)
There are about 16 of such algorithms.

No hetero backend uses-cases has been identified yet.


## Suggestion: Pass an algorithm hint to the backend

Pass an algorithm hint (a tag identifying the calling algorithm) from the "glue" level
(`glue_algorithm_impl.h`/`glue_numeric_impl.h`/`glue_memory_impl.h`) to the backend level
(`parallel_backend_tbb.h`/`parallel_backend_omp.h`/`parallel_backend_serial.h`).

It will be up to the backend to decide whether to use the hint, and how to use it.

### Why not just use the existing grain size argument?

The glue level does not call the backend directly; it goes through a middle pattern layer
(`__pattern_walk1`/`__pattern_walk2`/`__pattern_walk3`, `__parallel_reduce`, etc.) shared by many
unrelated algorithms, where the information about the original algorithm is lost.
So the grain size argument must be passed at the glue level via all layers.
This low-level information should not be exposed there, expecially because it is backend-specific,
e.g. SYCL backend does not have it.

An algorithmic tag may also be more versatile than a plain grain size: it could serve as a hint for
choosing a different partitioner, for example. This is speculative, and its benefit is not yet proven.

There is a single case of passing a grain size:
`__merge_path_out_lim` passes a fixed `__merge_path_cut_off` (2000).
Other than that, it is not used.

### Complexity

The changes are mostly mechanical, but they touch a lot of files and interfaces.
The summary below counts the changes at the callee side (function signatures that
would gain a hint parameter) and the caller side (call sites that would need to supply/forward it).

"Glue" layer:
- `glue_algorithm_impl.h`: 64 call sites (54 distinct patterns called)
- `glue_numeric_impl.h`: 7 call sites (3 distinct)
- `glue_memory_impl.h`: 20 call sites (10 distinct)
- `glue_algorithm_ranges_impl.h`: 73 call sites (46 distinct)
- `glue_numeric_ranges_impl.h`: 6 call sites (2 distinct)
- `glue_memory_ranges_impl.h`: 6 call sites (6 distinct)

"Middle" pattern layer:
- `algorithm_impl.h`: 114 callee signatures (57 distinct names) / 73 backend calls (12 distinct)
- `algorithm_ranges_impl.h`: 79 callee signatures (39 distinct) / 1 backend call
- `numeric_impl.h`: 10 callee signatures (3 distinct) / 5 backend calls (4 distinct)
- `memory_impl.h`: 2 callee signatures / 0 backend calls
- `memory_ranges_impl.h`: 12 callee signatures (6 distinct) / 0 backend calls
- `hetero/algorithm_impl_hetero.h`, `hetero/algorithm_ranges_impl_hetero.h`,
  `hetero/numeric_impl_hetero.h`, `hetero/numeric_ranges_impl_hetero.h`,
  `hetero/memory_impl_hetero.h`, `hetero/memory_ranges_impl_hetero.h`: out of scope
  for now, since no hetero backend use case has been identified (see Motivation);
  for reference, they contain 61/62/10/5/2/6 callee signatures respectively feeding
  the SYCL backend.

"Backend" layer:
- `parallel_backend_tbb.h`: 9 entry points
- `parallel_backend_serial.h`: 9 entry points
- `parallel_backend_omp.h`: itself only includes `omp/*.h`; the actual 9 entry points
  are defined there, across 18 signatures (some have a vectorized/non-vectorized overload)

Total for the host backends (TBB, OpenMP, serial): 27 backend entry points (36 counting
the OpenMP overloads), ~107 distinct pattern-layer functions (217 signatures counting
overloads), and 176 glue-layer call sites - on the order of 300-400 signatures/call
sites overall, depending on whether overloads are counted individually.

### Questions

**Is it worth the effort as stated? Is it possible to do a more localized change?**

Probably not worth the effort,
until the performance tuning potential is proven for large sizes, not only small sizes.
Or that the performance with small sizes can be improved significantly and for the majority of algorithms.

For example, a more solid proof is needed that
algorithms using `__parallel_find` and `__parallel_or` (see the motivation)
benefit from grain size tuning for large sizes.
A table with the potential performance gains for small sizes and each algorithm can be added.
Then it can be analyzed for the impact of the improvement and
whether pattern-level grouping at the middle layer is possible.
