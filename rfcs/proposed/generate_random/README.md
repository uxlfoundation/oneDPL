# Bulk Random Number Generation: `generate_random`

## Introduction

This RFC proposes `oneapi::dpl::ranges::generate_random`, based on the C++26 API from
[P1068R11 "Vector API for random number generation"](https://wg21.link/P1068R11),
with execution-policy overloads for host parallelism and SYCL devices.
Related discussion: [oneDPL#1516](https://github.com/uxlfoundation/oneDPL/issues/1516).

Generating random values in bulk currently requires users to manage engine offsets,
vector widths, and output storage themselves to use oneDPL's parallel RNG facilities.
The proposed API replaces that code with a single algorithm call and lets the library
choose an efficient implementation for the target.

## Proposal

### Namespace and header

The declarations and examples use namespace `oneapi::dpl::ranges`, matching the
[oneDPL Parallel Range Algorithms specification][ranges-spec]. The initial release status
and namespace are subject to open question 4. The algorithm is declared in
`<oneapi/dpl/random>` and requires C++20 ranges support. The new engine and distribution
members extend the existing types in namespace `oneapi::dpl`.

### Public API

The four no-policy overloads and their customization behavior are specified in the
[C++26 working draft, `[alg.rand.generate]`][generate-random-spec]. They accept an output
range or an iterator/sentinel pair, an engine, and an optional distribution.

This RFC adds four policy-first overloads using policies from the
[oneDPL Execution Policies specification][policies-spec]. The extension's callable interface
is shown as function declarations below. Constraints are omitted from the API and member
declarations for brevity; range and vector-type support remain open questions.
`ExecutionPolicy` must be a oneDPL execution policy.

```cpp
namespace oneapi::dpl::ranges {
  // oneDPL execution-policy extensions
  template<class ExecutionPolicy, class R, class G>
    std::ranges::borrowed_iterator_t<R> generate_random(ExecutionPolicy&& exec, R&& r, G&& g);
  template<class ExecutionPolicy, class O, class S, class G>
    O generate_random(ExecutionPolicy&& exec, O first, S last, G&& g);
  template<class ExecutionPolicy, class R, class G, class D>
    std::ranges::borrowed_iterator_t<R>
    generate_random(ExecutionPolicy&& exec, R&& r, G&& g, D&& d);
  template<class ExecutionPolicy, class O, class S, class G, class D>
    O generate_random(ExecutionPolicy&& exec, O first, S last, G&& g, D&& d);
}
```

The algorithm fills the output, advances the engine and distribution state, and returns
the end iterator (or `std::ranges::dangling` for a non-borrowed temporary range).
Engine-only generation must match repeated calls to the engine; distribution generation
must preserve the requested random distribution.

Calls without an execution policy run on the calling thread. Host policies select
sequential or parallel execution where supported; device policies request SYCL offload.
Engines and distributions can provide bulk-generation customizations. Otherwise, host
calls fall back to elementwise generation. Device calls require a compatible customization.

Feature availability is reported by `ONEDPL_HAS_GENERATE_RANDOM`.

### Engine additions

Add bulk-generation members to `philox_engine` and `linear_congruential_engine`:

```cpp
template<class R>
  constexpr std::ranges::borrowed_iterator_t<R> generate_random(R&& r);

// philox_engine only in the initial scope
template<class ExecutionPolicy, class R>
  std::ranges::borrowed_iterator_t<R> generate_random(ExecutionPolicy&& exec, R&& r);
```

These members fill the output range with engine values and advance the engine state.
`ranges::generate_random(r, g)` selects the engine's member when available.

### Distribution additions

Add bulk-generation members to `uniform_real_distribution` and `normal_distribution`:

```cpp
template<class R, class G>
  std::ranges::borrowed_iterator_t<R> generate_random(R&& r, G&& g);
template<class ExecutionPolicy, class R, class G>
  std::ranges::borrowed_iterator_t<R> generate_random(ExecutionPolicy&& exec, R&& r, G&& g);
```

The policy-free member of `uniform_real_distribution` is additionally `constexpr`.
These members use the supplied engine to fill the output range according to the
distribution's parameters, updating the engine and any cached distribution state.
`ranges::generate_random(r, g, d)` selects the distribution's member when available.

For both engines and distributions, the policy-free members are standard customization
hooks; the policy-taking members are oneDPL extensions.

### Example

```cpp
#include <oneapi/dpl/execution>
#include <oneapi/dpl/random>
#include <vector>

using engine_t = oneapi::dpl::philox4x32;
engine_t g(777);
std::vector<engine_t::result_type> integers(1'000'000);
oneapi::dpl::ranges::generate_random(integers, g);

std::vector<float> values(1'000'000);
oneapi::dpl::uniform_real_distribution<float> d(0.f, 1.f);
oneapi::dpl::ranges::generate_random(values, g, d);
oneapi::dpl::ranges::generate_random(oneapi::dpl::execution::par_unseq, values, g, d);
```

For device output, pass a device policy and device-accessible storage:

```cpp
// q is a SYCL queue; usm points to n output elements accessible on its device.
oneapi::dpl::ranges::generate_random(oneapi::dpl::execution::make_device_policy(q),
                                    usm, usm + n, g);
```

A runnable example is available in [generate_random_example.cpp](generate_random_example.cpp).

### Initial scope

| Component | Sequential host | Parallel host | SYCL device |
| --- | --- | --- | --- |
| `philox_engine` | Yes | Yes | Yes |
| `linear_congruential_engine` | Yes | No | No |
| `uniform_real_distribution` | Yes | With a supporting engine | With a supporting engine |
| `normal_distribution` | Yes | Engine generation only | With a supporting engine |

Other engines and distributions use the host fallback. Host scalar generation is available
without a SYCL compiler; vector-valued engines still require SYCL.

The prototype currently accepts only sized, random-access output ranges. Supporting the
broader output ranges allowed by the standard remains an open item.

### Compatibility and expected benefit

The API is additive and preserves existing scalar RNG interfaces and behavior. This
header-only addition preserves engine and distribution object layouts and stream
serialization formats, introducing no ABI break or new library dependencies.

The expected benefit is simpler user code and faster bulk generation through vectorization,
host parallelism, and device offload. Performance measurements are needed before release.

### Alternatives considered

- **Provide only the standard overloads.** Pro: a smaller API matching the standard.
  Con: no explicit host parallelism or device offload, which are central to oneDPL's
  execution model.
- **Use a separate name for parallel generation.** Pro: clearly distinguishes the oneDPL
  extension from the standard API. Con: introduces another algorithm name for the same
  operation. Policy-first overloads keep one name and follow existing oneDPL conventions.

## Open Questions

To resolve at the **proposed** stage:

1. Should we support random-access ranges only or all ranges supported by the standard,
   including `std::list`?
2. Should we support distributions returning `sycl::vec`, particularly
   `normal_distribution<sycl::vec<T, N>>`?
3. Which compilers should support this functionality?
4. Should `generate_random` first be released as experimental in
   `oneapi::dpl::experimental::ranges` or directly as supported in `oneapi::dpl::ranges`?

To resolve before release:

5. What validation and performance criteria are required for supported status?
6. Which release should determine the `ONEDPL_HAS_GENERATE_RANDOM` macro value?

## Next Steps

1. Review the public API and resolve the scope questions.
2. Add repository tests for conformance, engine and distribution state, and execution-policy
   behavior; validate device generation on GPU hardware.
3. Measure performance against current oneDPL usage and oneMKL RNG, and add user documentation.

[generate-random-spec]: https://eel.is/c++draft/alg.rand.generate
[ranges-spec]:
  https://uxlfoundation.github.io/oneDPL/specification/parallel_api/parallel_range_api.html
[policies-spec]:
  https://uxlfoundation.github.io/oneDPL/specification/parallel_api/execution_policies.html
