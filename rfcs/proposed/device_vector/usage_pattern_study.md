# `device_vector` Usage Pattern Study

This document surveys real-world usage of `thrust::device_vector`,
`dpct::device_vector`, and alternative device vector implementations across
open-source projects. It serves as supporting evidence for the design
decisions in the [device_vector RFC](README.md).

## Thrust (`thrust::device_vector`) — Native CUDA Projects

GitHub code search reports ~2,520 files containing `thrust::device_vector`
across the platform, spanning AI/ML, HPC, scientific computing, robotics,
graph analytics, and databases. A survey of notable projects follows.

### Sparse BLAS (spblas-reference)

[spblas-reference](https://github.com/SparseBLAS/spblas-reference) (sparse
BLAS standard reference implementation) demonstrates the minimal-but-dominant
pattern: `device_vector` as an **RAII device memory manager and host-device
data shuttle**:

1. **Constructing from `std::vector`** (~90% of uses) — bulk host-to-device
   transfer at setup time.
2. **Allocating output buffers by size** — e.g. after a symbolic phase
   computes output NNZ, a `device_vector` is constructed with just a count.
3. **Extracting raw device pointers** via `.data().get()` — every
   `device_vector` is ultimately unwrapped to a raw pointer for passing
   to library APIs (`csr_view`, `std::span`).
4. **Copying results back to host** via `thrust::copy(d.begin(), d.end(),
   host.begin())` — used in every test for verification.

Notably absent from spblas: element-level access (`operator[]`), `resize()`,
`push_back()`, `insert()`/`erase()`, or device-side algorithms on iterators.

### AI/ML Projects

**Notable finding:**  A few high profile performance-sensitive AI/ML codebases have
**explicitly moved away from `thrust::device_vector`**, while other ML
projects remain heavy users.

- [CUTLASS](https://github.com/NVIDIA/cutlass) (NVIDIA, 9.5k stars) — Uses
  `device_vector` in **30 files**, but only in examples and tests as
  scaffolding. Pattern: construct from host data, extract raw pointer via
  `.data().get()`, pass to GEMM kernels. Never used in hot-path
  implementations.
- [FAISS](https://github.com/facebookresearch/faiss) (Meta, 39.7k stars) —
  **Rolled their own `DeviceVector<T>`** instead of using
  `thrust::device_vector`. Reasons cited: control over streams, avoiding
  unwanted `T()` initialization on `resize()`, and custom memory growth
  strategy (power-of-2 below 4MB, 1.25x to 128MB, exact above).
- [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) (23.8k stars) —
  Uses `device_vector` in **24 files** for GPU kernel implementations (mode,
  graph reindex, CTC align, kron). Pattern: temporary containers within GPU
  kernels combined with `thrust::sort_by_key`, `thrust::reduce_by_key`.
- [cuDF](https://github.com/rapidsai/cudf) (RAPIDS, 9.6k stars) —
  **Explicitly discourages `thrust::device_vector`** in developer guide.
  Recommends `rmm::device_uvector` for uninitialized allocation and
  stream-ordered operations.
- [cuML](https://github.com/rapidsai/cuml) (RAPIDS, 5.2k stars) — Only **4
  files** still using it. Developer guide discourages it in favor of
  `MLCommon::device_buffer<T>` for stream-safe allocation.
- [H2O4GPU](https://github.com/h2oai/h2o4gpu) (466 stars) — Heavy user
  (**17 files**) for K-means, GLM, TSVD, ARIMA. Notable patterns: arrays of
  `device_vector` pointers for multi-GPU (`thrust::device_vector<T>
  *centroid_dots[n_gpu]`), `thrust::inner_product()` for convergence checks,
  raw pointer extraction for cuBLAS/cuSOLVER calls.
- [Zoph_RNN](https://github.com/isi-nlp/Zoph_RNN) (185 stars) — Neural
  machine translation. Class members for softmax state (`thrust_d_outputdist`,
  `thrust_d_normalization`) paired with host vector counterparts.

### HPC / Scientific Computing / Graph Analytics

These domains remain the heaviest `thrust::device_vector` users:

- [Gunrock](https://github.com/gunrock/gunrock) (1k stars) — **49 files**,
  heaviest user among notable projects. Type alias
  `device_vector_t = thrust::device_vector<type_t>`. Used for BFS outputs,
  graph frontier data. Raw pointer via `.data().get()`.
- [AmgX](https://github.com/NVIDIA/AMGX) (NVIDIA, 662 stars) — Custom
  allocator wrapper using `cudaMallocAsync`/`cudaFreeAsync` for stream-ordered
  allocation: `thrust::device_vector<T, thrust_amgx_allocator<T>>`.
- [GPU-Voxels](https://github.com/fzi-forschungszentrum-informatik/gpu-voxels)
  (315 stars) — **29 files**, robotics collision detection. Class members for
  octree nodes, voxel lists. Tracks allocations via
  `thrust::device_vector<void*>`.
- [ISCE3](https://github.com/isce-framework/isce3) (204 stars) — SAR radar,
  **21 files**. Class members for satellite orbit data
  (`thrust::device_vector<Vec3> _position, _velocity`).
- [Feltor](https://github.com/feltor-dev/feltor) (38 stars) — Plasma physics.
  Type aliases as vocabulary types:
  `using DVec = thrust::device_vector<double>`.

### Consolidated Construction Patterns (ordered by frequency)

1. `thrust::device_vector<T> d_v = h_v;` (copy from host_vector)
2. `thrust::device_vector<T> d_v(N);` (sized, value-initialized)
3. `thrust::device_vector<T> d_v(N, val);` (sized with fill value)
4. `thrust::device_vector<T> d_v(ptr, ptr + N);` (from host pointer range)
5. `new thrust::device_vector<T>(N);` (heap-allocated, multi-GPU)

### Why AI/ML Projects Rejected `thrust::device_vector`

The reasons cited by FAISS, cuDF, and cuML for moving away are instructive
for our design:

1. **Unwanted value initialization** — `resize()` and sized construction
   zero-initialize elements via device kernel. For large temporary buffers
   this is wasted work. (Supports our open question on `no_init_t` tags.)
2. **No stream/queue parameter** — Operations are synchronous or use a
   default stream, preventing overlap with other work.
3. **Header includes device code** — Forces `.cu` compilation even for host
   code that just manages device_vectors.

## dpct (`dpct::device_vector`) — Migrated CUDA-to-SYCL Projects

A broader survey of `dpct::device_vector` usage across ~18 repositories
(111 code results on GitHub) shows additional patterns beyond the
spblas-minimal case:

**Projects surveyed include:**
- [HeCBench](https://github.com/ORNL/HeCBench) (ORNL, 285+ stars) — HPC
  benchmark suite
- [oneAPI-samples](https://github.com/oneapi-src/oneAPI-samples) (Intel,
  1139+ stars) — radix sort migration samples
- [SYCLomatic-test](https://github.com/oneapi-src/SYCLomatic-test) —
  official compatibility test suite
- Various sparse matrix, radio astronomy, and optimization codes

**Consolidated operation frequency:**

| Operation | Frequency | Example |
|---|---|---|
| Construction from size | Very high | `dpct::device_vector<T> v(N)` |
| Assignment from `std::vector` | Very high | `d_vec = h_vec` (H2D) |
| `begin()`/`end()` for algorithms | Very high | `sort(policy, dv.begin(), dv.end())` |
| `data()` + raw pointer extraction | Very high | `get_raw_pointer(dv.data())` for kernels |
| Copy back to host | High | `std::copy(policy, dv.begin(), dv.end(), h.begin())` or `h = d` |
| Construction from `std::vector` | High | `dpct::device_vector<T> dv(host_vec)` |
| `size()` | Medium | For bounds checks, kernel launch args |
| `operator[]` | Medium | Host-side element access |
| `resize()` | Medium | Output buffer sizing |
| As class/struct member | Medium | Sparse matrices, particle systems |
| `clear()`, `push_back()`, `insert()`, `erase()` | Low | Mostly in tests, not real workloads |

**Notable real-world patterns:**
- **Default-construct then assign** — oneAPI-samples shows a "reset and
  re-sort" loop: `dpct::device_vector<T> d_keys;` then `d_keys = h_keys;`
  each iteration.
- **Raw pointer extraction is ubiquitous** — nearly every project that passes
  data to SYCL kernels unwraps `device_vector` to a raw pointer. This
  validates our `device_pointer::get()` and the importance of making
  `device_pointer` device-copyable for direct kernel capture. It seems likely
  this is a product of cuda patterns, but removes all possibility of overhead
  for performance sensitive codes, so it makes sense.

## Alternatives Built by Projects That Rejected `thrust::device_vector`

Two notable alternatives were built by high-performance projects that found
`thrust::device_vector` insufficient. Understanding their designs is
instructive for our proposal.

### FAISS `DeviceVector<T>` (Meta, 39.7k stars)

[Source: `faiss/gpu/utils/DeviceVector.cuh`](https://github.com/facebookresearch/faiss/blob/main/faiss/gpu/utils/DeviceVector.cuh)

FAISS built a minimal replacement explicitly motivated by three deficiencies
in Thrust (from the class comment): *"has more control over streams, whether
resize() initializes new space with T() (which we don't want), and control
on how much the reserved space grows."* It is restricted to POD types only.

**Key design choices:**
- **Explicit `cudaStream_t` on every host mutating operation** — `resize(n, stream)`,
  `append(ptr, n, stream)`, `setAt(i, val, stream)`, `getAt(i, stream)`,
  `reserve(n, stream)`, `reclaim(exact, stream)`.
- **No initialization on `resize()`** — comment: *"Don't bother zero
  initializing the newly accessible memory (unlike thrust::device_vector)"*.
  However, newly allocated raw capacity *is* zeroed.
- **Tiered growth strategy** — power-of-2 below 4M elements, 1.25x up to
  128M elements, exact allocation above. Prevents overallocation for large
  buffers.
- **No iterators, no `operator[]`** — element access only via explicit
  `setAt()`/`getAt()` methods with stream parameter. No proxy references.
- **Auto-detects host vs device source** — `append()` uses
  `cudaPointerGetAttributes` to pick the right `memcpy` direction.
- **Custom memory resource** — allocates through FAISS's `GpuResources`
  abstraction (pool for temporaries, `cudaMalloc` for persistent data),
  not through CUDA allocator APIs directly.
- **No copy semantics** — move-only via the underlying RAII memory handle.

**What it strips out vs Thrust:** iterators, `operator[]`, `push_back`,
`insert`/`erase`, copy construction, implicit host↔device conversion,
value initialization, STL container compatibility. What it **adds:**
explicit stream parameter, `append()` with auto-direction detection,
`reclaim()` for capacity shrinking, and return values indicating whether
reallocation occurred.

### RAPIDS `rmm::device_uvector<T>`

[Source: `rmm/include/rmm/device_uvector.hpp`](https://github.com/rapidsai/rmm/blob/main/cpp/include/rmm/device_uvector.hpp)

RMM's `device_uvector` (the "u" stands for uninitialized) is the
recommended replacement for `thrust::device_vector` across the RAPIDS
ecosystem (cuDF, cuML, cuGraph). It was motivated by the same concerns
as FAISS plus a desire for pluggable memory resources.

**Key design choices:**
- **Explicit `cuda_stream_view` on every operation** — construction,
  resize, reserve, element access, copy construction all require a stream.
- **No initialization** — construction and resize never launch a kernel
  to zero-fill or value-initialize. This is the defining feature.
- **No geometric growth** — `resize()` allocates exactly the requested
  size. RMM's pool memory resources handle allocation performance, so
  container-level overallocation is unnecessary.
- **Deleted default and copy constructors** — must provide stream to
  construct. Copy requires explicit call: `device_uvector(other, stream)`.
- **Iterators are raw `T*` pointers** — usable in device code and thrust
  algorithms, but dereferencing on host is undefined behavior. No proxy
  references.
- **No `operator[]`** — element access via explicit `element(i, stream)`
  (synchronous D→H) and `set_element_async(i, val, stream)` (async H→D).
  The async setter deliberately deletes its rvalue overload to prevent
  dangling references.
- **No bulk host↔device transfer API** — no constructor from host data,
  no `assign()` from host range. Users must use `cudaMemcpy` directly.
- **Pluggable memory resource** — uses `device_async_resource_ref`
  (type-erased, `std::pmr`-style). Default is `cudaMalloc`, but can be
  pool, arena, etc.
- **`static_assert(is_trivially_copyable<T>)`** — only trivially copyable
  types.
- **Implicit conversion to `cuda::std::span<T>`** — lightweight view
  interop.
- **Stores device ID** — destructor deallocates on the correct device even
  if a different device is current.

**What it strips out vs Thrust:** `operator[]`, `push_back`, `insert`/`erase`,
`assign`, `clear`, `swap`, implicit copy, host range construction, value
initialization, geometric growth, non-trivial types. What it **adds:**
explicit stream everywhere, pluggable memory resource, span conversion,
device-aware destruction.

Note: RMM also provides `rmm::device_vector<T>`, which is just a type alias
for `thrust::device_vector<T, rmm::mr::thrust_allocator<T>>` — same Thrust
interface but with RMM-backed allocation.

## Summary
Most usage seems to focus on `device_vector` as a convenient way to allocate and control lifetime of device memory.
Usage largely focuses on:
 * copies to and from host side vector all at once
 * getting raw pointers to use directly on the device
 * using begin() and end() iterators as input to algorithms

Host side usage
 * Largely not present
 * If present, mostly used in tests / useful in debugging
 * dpct migrations use `operator[]` in some cases, but this may be from CUDA migration patterns rather than intentional
 * Cases which do need host access (FAISS, RMM) have replaced device_vector with alternatives that allow access with explicit stream synchronization.

