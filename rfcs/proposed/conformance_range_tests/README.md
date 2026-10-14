# Conformance Tests for oneDPL Range Algorithms

## Introduction

oneDPL provides parallel range algorithms in `oneapi::dpl::ranges`: C++20 constrained algorithms with an
execution policy as the first argument, for host policies (`seq`, `unseq`, `par`, `par_unseq`) and device policies.
Like the `std::ranges` algorithms, each of them is constrained by a *requires-clause* (`std::indirectly_copyable`,
`std::sortable`, `std::indirect_unary_predicate`, ...). That clause is the whole contract with the user: a program
whose types satisfy it must compile and must produce the specified result.

The existing tests in `test/parallel_api/ranges/std_ranges_*.pass.cpp` use convenient types: `int`, `std::vector`,
`std::span`, aggregates with all special members, predicates taking `const T&` or `auto`. Such types provide much
more than the requirements promise, so an implementation that uses an operation outside the contract still passes.
For example:
- copy-constructing an element where `std::indirectly_copyable` grants only `*out = *in`;
- default-constructing a buffer where `std::sortable` grants only moves;
- passing a const lvalue to a predicate that the concept invokes only with `iter_reference_t` (here `T&`);
- calling `r.size()` or `r[i]` on a range that does not provide them, or using `end(r)` as an iterator;
- taking an address with `&x` instead of `std::addressof(x)`.

Each of these compiles with `int`, but fails for a conforming user type. The same tests also use one value type and
projections that coincide for both sequences, so applying a projection to the wrong sequence goes unnoticed.

This RFC describes a test suite that closes both gaps. It is implemented in
[PR #2832](https://github.com/uxlfoundation/oneDPL/pull/2832) (branch `dev/skopienko/range_conformance_tests`).
The suite has already found defects in the library. Some are fixed, for example `__simd_find_first_of()`
([PR #2834](https://github.com/uxlfoundation/oneDPL/pull/2834)) and the vectorized `__brick_lexicographical_compare`,
which applied the first projection to the second sequence. The others are recorded as known gaps (see
[Tracking Known Implementation Gaps](#tracking-known-implementation-gaps)).

The goals are:
1. Every range algorithm compiles and gives the right result with types that provide *exactly* what its
   requires-clause demands, with every host policy and with a device policy.
2. Each projection of a two-range algorithm is applied to its own sequence.
3. Known deviations from the contract are visible, attributed to an algorithm and a policy group, and removed
   together with the library fix, instead of being worked around in the tests.

## Proposal

The proposal has three parts:
- an **archetype conformance suite** in a new directory `test/parallel_api/ranges/conformance/`;
- **projection checks** added to the existing two-range tests;
- **gate macros** in `test/support/test_config.h` for the cases that currently fail.

### Layout

| File | Contents |
|---|---|
| `conformance/std_ranges_archetypes_base.h` | archetype iterator, sentinel and views; storage; helper macros |
| `conformance/std_ranges_archetypes_{read,value,write,permute,merge,storable,memory}.h` | element and callable archetypes per requirement family |
| `conformance/std_ranges_archetypes.h` | includes all of the above |
| `conformance/std_ranges_algo_archetypes_test.h` | test driver for the algorithm families |
| `conformance/std_ranges_memory_archetypes_test.h` | test driver for the memory algorithms over `archetype_view` |
| `conformance/std_ranges_algo_archetypes_{read,value,cross,write,permute,merge,storable}.pass.cpp`, `conformance/std_ranges_memory_archetypes.pass.cpp` | the eight test targets |
| `std_ranges_test.h` and eleven `std_ranges_*.pass.cpp` | projection checks, unique kernel names |
| `std_ranges_memory_test.h`, `std_ranges_partial_sort_copy.pass.cpp`, `std_ranges_rotate_copy.pass.cpp` | smaller changes to the existing tests |
| `test/support/test_config.h` | 33 gate macros |

### Archetype Conformance Suite

#### Principles

- **Exactly the requirements.** An *archetype* is a type that satisfies one requirement family and nothing more.
  Everything else is deleted or left undeclared: default construction, copy and move (unless granted), comparison
  operators (unless granted), and the unary `operator&`.
- **Real calls.** Coverage comes only from calling the algorithm, so that its body is instantiated with the
  archetypes. A `static_assert` on a concept proves only that the declaration is constrained correctly.
- **Every policy.** Each call runs with `seq`, `unseq`, `par` and `par_unseq`, and with a device policy when the
  SYCL backend is available. The host and device bodies of an algorithm are different code.
- **Self-checking archetypes.** Every archetype header has `static_assert`s on both sides: the concepts the type must
  satisfy, and the properties it must *not* have (`!std::copy_constructible`, `!std::default_initializable`,
  `!std::totally_ordered`, ...). An archetype therefore cannot silently become too permissive.
- **Results, not only compilation.** Each call has a checker that verifies the returned iterator or value and the
  content of the output where this is meaningful.

#### Ranges and Storage

`std_ranges_archetypes_base.h` provides:
- `archetype_iterator<T>`: a random-access iterator over `T*` with `reference = T&`. It is not contiguous.
- `archetype_sentinel<T>`: a distinct sentinel type that satisfies `std::sized_sentinel_for`.
- `archetype_view<T>`: random-access, sized, borrowed, neither contiguous nor common. It derives from
  `std::ranges::view_interface`.
- `plain_archetype_view<T>`: the same range *without* `view_interface`, so `size()`, `operator[]`, `empty()` and
  `front()` are absent. It catches code that calls these members instead of `std::ranges::size()` and the iterators.
- `archetype_storage<T, Alloc>`: allocates raw memory and constructs each element in place from a factory with
  `std::construct_at`. Elements therefore need neither default construction nor copy. The storage uses
  `std::allocator` for host policies and `sycl::usm_allocator<T, sycl::usm::alloc::shared>` for a device policy.

The `static_assert`s at the end of the header pin all these properties down.

The ranges are not common, so a correct algorithm must compute the end iterator as `begin(r) + size(r)`. They are
not contiguous, so pointer-based shortcuts are not taken.

#### Element and Callable Archetypes

Common building blocks:
- `TEST_ARCHETYPE_DELETED_OPERATIONS(T)` deletes the copy and move constructors and assignments.
  `TEST_ARCHETYPE_DEFAULTED_OPERATIONS(T)` defaults them. Both delete `operator&`; this can be turned off with
  `TEST_ARCHETYPE_DELETE_ADDRESSOF=0`.
- **Device-copyable twins (`*_dc`).** A device policy needs trivially copyable, device-copyable element types, so
  most host archetypes have a `_dc` twin. It provides the same constructors, assignments and comparisons, but with
  the special members defaulted. `TEST_ARCHETYPE_CHECK_DEVICE_COPYABLE` asserts `std::is_trivially_copyable_v` and
  `sycl::is_device_copyable_v`. On the device the suite therefore checks everything except "non-copyable" and
  "move-only".
- **Mutable callables (`*_mut`).** Predicates, comparators, operations and projections whose parameters are
  non-const lvalue references, for example `bool operator()(read_archetype&, read_archetype&) const`. They satisfy
  the concepts because the iterator's reference is `T&`. They are asserted *not* invocable with const arguments, so
  any const injection fails to compile: a `const auto&` local, a const view, a `const T&` brick parameter, a const
  copy of an element.
- **Default operations.** Elements with only `==`, or with `==` and `<=>`, check the default `ranges::equal_to` and
  `ranges::less` without requiring more than these operators.
- **Companion types.** Member aliases (`out_type`, `value_arg`, `nocopy_value_type`) let one generic call lambda
  pick the matching output or value type for the host and the device element.
- **Cross-type equality.** `std::common_type` specializations make `ranges::equal_to` valid between an element and
  an unrelated value type (`search_value`). They are needed for `find`, `count`, `remove`, `replace`, ...
  `[namespace.std]` allows specializing `common_type` for program-defined types.

#### Test Driver

`std_ranges_algo_archetypes_test.h` runs a `call` lambda on fresh storage of `archetype_test_size = 1000` elements.
By default element `i` holds the value `i`. It then passes the views and the result to a `check` lambda.

| Function | Ranges |
|---|---|
| `run_algo<E>`, `run_algo_plain<E>` | one `archetype_view` / `plain_archetype_view` |
| `run_algo2<E1, E2>`, `run_algo2_plain<E1, E2>` | two ranges with the same values |
| `run_algo2_offset<E1, E2>` | the second range holds `i + 1000`, so an output's initial content differs from the input |

`run_algo`, `run_algo2` and `run_algo2_offset` come in three wrappers; `run_algo_plain` and `run_algo2_plain` only
in the third one:
- `*_host_policies` runs `seq`, `unseq`, `par` and `par_unseq`.
- `*_hetero_policies<..., CallId>` runs `TestUtils::get_dpcpp_test_policy<CallId>()`.
- `*_all_policies<E, E_dc, CallId>` runs both, with the host element on the host and the `_dc` element on the device.

`CallId` must be unique within a translation unit, because the policy name becomes part of the SYCL kernel names
(`-fno-sycl-unnamed-lambda`).

```cpp
// reverse_copy from a non-copyable input into an output that is assignable only from a non-const input lvalue
run_algo2_all_policies<copy_in_archetype, copy_out_archetype, copy_in_archetype_dc, copy_out_archetype_dc, 2>(
    [](auto&& policy, auto&& in_view, auto&& out_view) {
        return dpl_ranges::reverse_copy(std::forward<decltype(policy)>(policy), in_view, out_view);
    },
    [](auto&& in_view, auto&& out_view, auto) {
        const auto n = std::ranges::size(in_view);
        return std::ranges::begin(out_view)[0].val == std::ranges::begin(in_view)[n - 1].val &&
               std::ranges::begin(out_view)[n - 1].val == std::ranges::begin(in_view)[0].val;
    },
    "reverse_copy");
```

When a call needs an extra output, for example `partition_copy`, `merge` or a binary `transform`, the call lambda
creates it with `make_out_storage` and verifies the result itself.

#### Test Families

| Target | Requirement family | Algorithms | Calls |
|---|---|---|---|
| `..._read` | `indirectly_unary_invocable`, `indirect_unary_predicate`, `indirect_strict_weak_order`, `indirect_binary_predicate` over immovable elements | `for_each`, `find_if`, `find_if_not`, `find_last_if`, `find_last_if_not`, `any_of`, `all_of`, `none_of`, `is_partitioned`, `count_if`, `min_element`, `max_element`, `minmax_element`, `is_sorted`, `is_sorted_until`, `is_heap`, `is_heap_until`, `adjacent_find` | 50 |
| `..._value` | `indirect_binary_predicate<ranges::equal_to, projected<I, Proj>, const T*>` with a value of an unrelated type; `permutable` for `remove` | `find`, `find_last`, `count`, `contains`, `search_n`, `remove` | 18 |
| `..._cross` | `indirectly_comparable<I1, I2, Pred>` and `indirect_strict_weak_order<Comp, I1, I2>` over two unrelated element types | `equal`, `mismatch`, `search`, `find_end`, `find_first_of`, `includes`, `contains_subrange`, `starts_with`, `ends_with`, `lexicographical_compare` | 31 |
| `..._write` | `indirectly_writable`, `indirectly_copyable`, `indirectly_movable`, `indirectly_swappable`, writable results of `transform` | `fill`, `copy`, `reverse_copy`, `rotate_copy`, `move`, `swap_ranges`, `transform` (unary and binary, with and without projections), `copy_if`, `remove_copy`, `remove_copy_if`, `unique_copy`, `partition_copy`, `replace`, `replace_if`, `replace_copy`, `replace_copy_if` | 33 |
| `..._permute` | `permutable` over move-only elements, `sortable<I, Comp, Proj>` | `reverse`, `rotate`, `shift_left`, `shift_right`, `remove_if`, `unique`, `partition`, `stable_partition`, `sort`, `stable_sort`, `is_sorted`, `partial_sort`, `partial_sort_copy`, `nth_element`, `inplace_merge` | 35 |
| `..._merge` | `mergeable<I1, I2, O, Comp>` | `merge`, `set_union`, `set_difference`, `set_intersection`, `set_symmetric_difference` | 11 |
| `..._storable` | `indirectly_copyable_storable` (the value-returning `min`, `max`, `minmax`) | `min`, `max`, `minmax` | 9 |
| `std_ranges_memory_archetypes` | the construction kind each memory algorithm requires | `uninitialized_default_construct`, `uninitialized_value_construct`, `uninitialized_fill`, `uninitialized_copy`, `uninitialized_move`, `destroy` | 9 |

Where it applies, each algorithm is called with const callables, with `_mut` callables, with the default
predicate or comparator, and with a projection. A few calls go through `plain_archetype_view`.

The archetypes that carry the important restrictions are:
- **read and cross:** elements cannot be copied, moved, compared or have their address taken, so every comparison
  goes through the user's callable. `cross_pred` exists only for the (lhs, rhs) argument order, so swapped
  arguments are a compile error. `cross_comp` has all four orders that `std::strict_weak_order` requires.
- **value:** the searched value has its own type. On the host it is non-copyable and immovable as well, and
  equality across the two types is valid only through `common_type`.
- **write:**
  - `copy_out_archetype` is assignable only from a *non-const* `copy_in_archetype&` and cannot be constructed from
    it.
  - `move_out_archetype` is assignable only from an rvalue.
  - `swap_archetype` is swappable only through its ADL `swap`.
  - `transform` outputs accept only the operation's result type, and the projected operations accept only the
    projection's result type, so a skipped projection does not compile.
- **permute:** elements are move-only, with user-provided moves and no default constructor.
  `partial_sort_copy` uses an input that is not movable at all, an output that is assignable from a non-const
  input lvalue, and a comparator over all four (in, out) combinations.
- **merge:** outputs are assignable only from a non-const input lvalue and cannot be constructed from an input.
- **memory:**
  - each constructor writes one field of a two-field object over memory filled with `0xFF`, so the checker can see
    which constructor ran (value-initialization zeroes both fields, default-initialization only one);
  - `copy_archetype` is constructible only from `transfer_source&`, `move_archetype` only from `transfer_source&&`;
  - `destroy_archetype` has a destructor and no usable constructor.

#### What the Checks Verify

The input is the identity sequence `0, 1, ..., 999`. The checkers verify returned iterators against
`begin(r) + k` rather than `end(r)`, returned counts, and selected output elements: `copy_if` keeps the multiples
of 3, the `*_copy` algorithms write `(n + 2) / 3` or `n - (n + 2) / 3` elements, `reverse` and `rotate` move
specific elements, `min` and `max` return the expected values, `partial_sort_copy` with the descending comparator
writes `999 ... 990`.

The suite targets the *requirements*. Detailed functional coverage stays with the existing `std_ranges_*` tests;
see [Limitations](#limitations).

### Projection Checks in the Existing Tests

The two-range tests now check that each projection is applied to its own sequence, on two levels.

**Compile level.** `std_ranges_test.h` gets two distinct element types and two projections that accept only their
own type:

```cpp
struct A { int a; operator int() const { return a; } };
struct B { int b; operator int() const { return b; } B& operator=(const A& a) { b = int(a); return *this; } };
auto proj_a = [](const A& a) { return a.a; };
auto proj_b = [](const B& b) { return b.b; };
```

`check_mixed_types_in_in_host` and `check_mixed_types_in_in_device` call `algo(policy, r1, r2, args...)` with
`r1` of `A` and `r2` of `B`, passing `proj_a` and `proj_b`. The host helper covers the four host policies; the
device helper uses USM shared memory and returns early if the device does not support it. The
`check_mixed_types_in_in_out_{host,device}` helpers do the same for algorithms with an output. A projection applied
to the wrong sequence is a compile error.

The `result_as_is`, `result_index`, `result_subrange` and `result_indices` reducers turn the different return types
into a `std::vector<int>`, so that the result can be compared with the serial checker's result.

The algorithms covered are `equal`, `mismatch`, `search`, `find_end`, `find_first_of`, `starts_with`, `ends_with`,
`contains_subrange`, `merge` and `transform`.

**Data level.** The data is generated so that the result is correct only if each projection reaches its own
sequence. One sequence is generated as `i`, the other as `2 * i` or `4 * i + 2`, and the projection makes them
match. This is done for `equal`, `find_end` and `find_first_of`. `lexicographical_compare` gets two cases:
- a "hidden pair", where the first elements are equivalent only under the swapped projections, which would hide the
  real first mismatch;
- element-wise equal projected sequences, where only the tie-break by size decides.

### Other Changes to the Existing Tests

- **Unique kernel names.** In `std_ranges_test.h`, the real call uses `CLONE_TEST_POLICY_IDX(exec, 0)`, and the
  return-type check for dangling iterators uses `CLONE_TEST_POLICY_IDX(exec, 1)`. Before, both used the same policy
  name, so an `auto`-returning tester that instantiated device code in the check produced two kernels with one name.
- **`std_ranges_rotate_copy.pass.cpp`.** The tester declares its return type explicitly, so the dangling check does
  not instantiate the device code.
- **`std_ranges_memory_test.h`.** `test_memory_algo` gets `call_id` (for kernel names) and `OutElem` (the output
  element type of `uninitialized_copy` and `uninitialized_move`) template parameters. Both have defaults, so the
  existing tests are unchanged.
- **`std_ranges_partial_sort_copy.pass.cpp`.** `test_projections_consistency()` uses two projections that order the
  input and the output differently, and compares every policy with `seq`. These projections violate a precondition
  of `partial_sort_copy`, so the test is to be replaced (T1 in
  [Classification of the Gated Calls](#classification-of-the-gated-calls)).

### Tracking Known Implementation Gaps

A call that does not compile, or returns a wrong result, is not removed and not weakened. It is compiled out by a
**gate macro** defined to `1` in `test/support/test_config.h`. The macro is removed together with the library fix.

**Naming.**
- `_TEST_CPP20_RANGES_BROKEN_REQUIRES_<ALGO>[_HOST|_HETERO]`: the algorithm needs more than its requires-clause; the
  archetype call does not compile.
- `_TEST_CPP20_RANGES_BROKEN_WRONG_RESULT_<ALGO>_<WHAT>`: the call compiles but gives a wrong result.
- The suffix names the broken side. `_HOST` means the four host policies, `_HETERO` the device policy, and no suffix
  means both.
- The macros are sorted by name. Gates are used only by the conformance tests and by the projection checks.

**Idioms.**

One side broken: only that half of the call is gated.

```cpp
// device half broken
{
    auto call = ...;
    auto check = ...;
    run_algo_host_policies<Elem>(call, check, "is_heap");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_IS_HEAP_HETERO
    run_algo_hetero_policies<ElemDc, 35>(call, check, "is_heap");
#endif
}

// host half broken; the outer condition keeps the lambdas out when neither half is compiled (-Werror)
#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_MIN_HOST || TEST_DPCPP_BACKEND_PRESENT
{
    auto call = ...;
    auto check = ...;
#    if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_MIN_HOST
    run_algo_host_policies<Elem>(call, check, "min");
#    endif
#    if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<ElemDc, 3>(call, check, "min");
#    endif
}
#endif // !_TEST_CPP20_RANGES_BROKEN_REQUIRES_MIN_HOST || TEST_DPCPP_BACKEND_PRESENT
```

Both sides broken: one unsuffixed macro and one merged call, `#if !MACRO` around `run_algo*_all_policies`. When an
algorithm has gaps on both sides, even in different test files, the gates are collapsed into one unsuffixed
macro. One gate per defect is easier to maintain than per-side bookkeeping.

#### Classification of the Gated Calls

There are 33 gates (32 `_REQUIRES_` and one `_WRONG_RESULT_`) over 60 gated calls. The tables below classify every
call by what fails and why. They describe the head of PR #2832 at `adaabb332`, which contains `main` as of
2026-10-13.

**Method.** Each gated call is compiled on its own: a copy of its test file has only that gate set to `0` (with the
outer guard of the host idiom), and every other gate stays `1`. A copy per call is needed because Clang reports an
error in a given template specialization only once, so with several gates open one failing call hides the next.
Each copy is compiled with `-fsyntax-only -std=c++20` in the modes its gate can affect:
- host side: the TBB, OpenMP and serial backends, each with all four host policies;
- device side: `-fsycl`; for an unsuffixed gate also `-fsycl` with the host half of the driver switched off, so
  that the device half is judged alone;
- compilers and standard libraries: icpx 2026.1 on Windows with the Microsoft STL (14.51) and on Linux with
  libstdc++ 13; on Linux the host side also with g++ 13.

Every error is attributed to the first oneDPL frame outside the generic wrappers in `utils.h` (`__binary_op`,
`__unary_op`, ...). Where that frame only passes the problem on, the tables cite the line that reading the code
shows as the cause, for example `utils.h:170`, `:451` or `:1137`. Every test target compiles cleanly in every mode
with all gates in place. The wrong-result gate was checked at run time by a probe that repeats the calls of its
test and prints the result of every policy, on Linux with a CPU device and on Windows with a CPU and a GPU device.

Limits of the method: a host cell fails if any of the four host policies fails; MSVC `cl`, Clang and libc++ were not
measured; and an error can hide another one behind it, for example in a function template that overload resolution
already rejected. A fix of one class can therefore expose the next.

**Defect classes.** `D` classes break only the device policy, `H` classes only the host policies (some of the
cited lines are in code that both sides share), and `T` is a test defect. Paths are relative to
`include/oneapi/dpl/pstl/`.

| Class | What the implementation does | Where | What the requirements grant instead |
|---|---|---|---|
| D1 | Passes a `const` element to the user's callable. Device input is staged read-only (`all_view` with `access_mode::read` yields `const T`) or bound to `const` locals and parameters. | `hetero/algorithm_impl_hetero.h:1081` (`is_partitioned`), `:1125` (`is_heap`, `is_heap_until`); `hetero/dpcpp/parallel_backend_sycl_reduce_then_scan.h:475`, `:515` and `hetero/dpcpp/unseq_backend_sycl.h:122`, `:557` (`partition`, `stable_partition`); `hetero/dpcpp/unseq_backend_sycl.h:632`, `:636` (`find_first_of`: a `const` copy of the element); `hetero/utils_hetero.h:138-141` (`lexicographical_compare`: `auto const&` locals); `hetero/dpcpp/parallel_backend_sycl_reduce_then_scan.h:666-678` (set algorithms); `hetero/dpcpp/parallel_backend_sycl_merge.h:131`, `:188` (`inplace_merge`) | `indirect_unary_predicate`, `indirect_strict_weak_order` and `indirect_binary_predicate` invoke the callable with `iter_reference_t` (`T&`) and `iter_value_t&`, never with `const T&` |
| D2 | Assigns the output from a `const` input element. | `hetero/dpcpp/unseq_backend_sycl.h:885` (`rotate_copy`); `hetero/algorithm_impl_hetero.h:398` (`__brick_copy` in `partial_sort_copy`) | `indirectly_copyable<In, Out>` grants `*out = *in` with `*in` of type `iter_reference_t<In>` |
| D3 | Assigns the output from a temporary or a `const` source. `static_cast<_ConvertedTupleType>(...)` creates a prvalue, which `__pstl_assign` takes as `const _Xp&`; `__pstl_assign` also receives the read-only first element in `unique_copy`; `partition_copy` assigns the prvalue of `__tuple_type_cast(...)` directly. | `hetero/dpcpp/parallel_backend_sycl.h:298`, `hetero/dpcpp/parallel_backend_sycl_reduce_then_scan.h:164`, `:182` (`copy_if`, `remove_copy`, `remove_copy_if`, `unique_copy`); `:2153` (`unique_copy`); `:238`, `:262` (`partition_copy`); `utils.h:170` (`__pstl_assign`) | the same as D2 |
| D4 | Wraps the range into `std::ranges::reverse_view` and indexes it through a `const` reference. A `const` `reverse_view` has `begin()`, and therefore `operator[]`, only over a common range. | `glue_algorithm_ranges_impl.h:203` (`find_last_if`; `find_last` and `find_last_if_not` forward to it); `hetero/dpcpp/unseq_backend_sycl.h:557` | `random_access_range` and `sized_range`; `common_range` is not required |
| D5 | Narrows the size of the range in a braced initializer, `__reverse_functor{__n}`. With libstdc++ the size of a C++20 `view_interface` range is signed. | `hetero/algorithm_ranges_impl_hetero.h:916` | `sized_range` gives `range_size_t`, which may be signed |
| D6 | Takes the address of an element with the unary `&`. | `utils.h:1137` (`__lazy_ctor_storage::__setup`, reached from the device set algorithms through `hetero/dpcpp/parallel_backend_sycl_reduce_then_scan.h:67`); `hetero/dpcpp/parallel_backend_sycl_radix_sort_one_wg.h:123`, `:311` (`sort` and `stable_sort` with an arithmetic key) | no concept requires a usable unary `&`; `std::addressof` is the portable spelling |
| H1 | Constructs output elements in raw memory from the input, instead of assigning them. | `set_algorithms_utils.h:91`, `memory_impl.h:96`, `:111` (set algorithms); `algorithm_impl.h:2707` (`partial_sort_copy` with `par` and `par_unseq`) | `mergeable` and `indirectly_copyable` grant assignment, not construction of the output value type from the input |
| H2 | The OpenMP backend keeps the merge buffer of its stable sort in a `std::vector<T>(n)`, which default-constructs. | `omp/parallel_stable_sort.h:107` | `sortable` grants moves only |
| H3 | Passes a `const` element to the comparator in the host parallel and vector code. `std::upper_bound` and `std::lower_bound` take the searched value as `const T&`; some lambdas and combiners take `const` parameters. | `parallel_backend_tbb.h:1023-1037`, `:1240-1245`, `omp/parallel_merge.h:48`, `:53` (merges in `sort`, `stable_sort`, `partial_sort`, `partial_sort_copy`, `inplace_merge`); `algorithm_impl.h:2841` (`nth_element`); `algorithm_impl.h:3303` (`std::inplace_merge` called with a range comparator); `unseq_backend_simd.h:643-755` (`min`, `max`, `minmax`) | the same as D1 |
| H4 | oneDPL calls `std::ranges::stable_sort` as the leaf sort of `stable_sort` with every host policy (on the whole range for `seq` and `unseq`, on each chunk for `par` and `par_unseq`), and `std::ranges::inplace_merge` in the serial `inplace_merge`. libstdc++ 13 implements both on top of its classic algorithms and calls the comparator with `const` lvalues; the Microsoft STL does not. | `glue_algorithm_ranges_impl.h:578`, reached through `algorithm_ranges_impl.h:453-455` and `algorithm_impl.h:2544`, `:2555-2558`; `algorithm_ranges_impl.h:847` | the same as D1; the defect is in the standard library, and oneDPL inherits it |
| H5 | The serial `inplace_merge` returns `__end(__r)` as `borrowed_iterator_t<R>`. For a range that is not common, that is the sentinel, which does not convert. | `algorithm_ranges_impl.h:848` | `bidirectional_range`; `common_range` is not required |
| H6 | `__replace_copy_functor` stores a copy of the new value (`const _Tp _M_value`), which needs a copyable `_Tp`. With a value that is not copyable, the functor is not copyable either, and every walk copies the functor. With g++ the first error comes earlier: `__ref_or_copy<_ExecutionPolicy, const _T>{__new_value}` is a braced cast to a reference type, which g++ 13 evaluates by copying. | `utils.h:451`, `:455`, created at `algorithm_ranges_impl.h:1754`; `glue_algorithm_ranges_impl.h:1182` | `indirectly_writable<O, const T&>` does not make `T` copyable |
| T1 | The test uses projections that order the input and the output differently (`-x` and `x`). That violates the precondition of `partial_sort_copy` that writing a value into the output does not change how `comp` with `proj1` or `proj2` orders it ([partial.sort.copy]/4 in the working draft; C++20 has no such paragraph). The parallel and device patterns rely on that precondition: they apply only `proj2`, after copying. | test: `std_ranges_partial_sort_copy.pass.cpp:78`; library: `algorithm_ranges_impl.h:510`, `hetero/algorithm_ranges_impl_hetero.h:1533` | the precondition makes `proj1` and `proj2` interchangeable for ordering |

**Gates.** "Host" is the outcome with the four host policies, "device" the outcome with the device policy; a dash
means that the gate does not cover that side.

| Gate | Gated calls | Host | Device | Classes |
|---|---|---|---|---|
| `COPY_IF_HETERO` | `write:150`, `write:343` | — | fails | D3 |
| `FIND_FIRST_OF_HETERO` | `cross:146` | — | fails | D1 |
| `FIND_LAST_HETERO` | `value:47`, `value:98`, `value:156` | — | fails | D4 |
| `FIND_LAST_IF_HETERO` | `read:58`, `read:206` | — | fails | D4 |
| `FIND_LAST_IF_NOT_HETERO` | `read:73`, `read:221` | — | fails | D4 |
| `INPLACE_MERGE` | `permute:172`, `permute:309`, `permute:372` | all backends; for `permute:172` and `:372` through `seq` only | fails only for `permute:309` | H5; H3, H4 and D1 for `permute:309` |
| `IS_HEAP_HETERO` | `read:308` | — | fails | D1 |
| `IS_HEAP_UNTIL_HETERO` | `read:320` | — | fails | D1 |
| `IS_PARTITIONED_HETERO` | `read:251` | — | fails | D1 |
| `LEXICOGRAPHICAL_COMPARE` | `cross:177` | **compiles** | fails | D1 |
| `MAX_HOST` | `storable:71` | all backends | — | H3 |
| `MINMAX_HOST` | `storable:89` | all backends | — | H3 |
| `MIN_HOST` | `storable:55` | all backends | — | H3 |
| `NTH_ELEMENT_HOST` | `permute:300` | all backends | — | H3 |
| `PARTIAL_SORT_COPY` | `permute:150`, `permute:276` | all backends | fails | H1, H2, D2; H3 for `permute:276` |
| `PARTIAL_SORT_HOST` | `permute:141`, `permute:267`, `permute:355` | OpenMP; TBB too for `permute:267` | — | H2; H3 for `permute:267` |
| `PARTITION_COPY_HETERO` | `write:202`, `write:402` | — | fails | D3 |
| `PARTITION_HETERO` | `permute:208` | — | fails | D1 |
| `REMOVE_COPY_HETERO` | `write:241`, `write:444` | — | fails | D3 |
| `REMOVE_COPY_IF_HETERO` | `write:166`, `write:362` | — | fails | D3 |
| `REPLACE_COPY_HOST` | `write:279`, `write:484` | all backends | — | H6 |
| `REPLACE_COPY_IF_HOST` | `write:258`, `write:462` | all backends | — | H6 |
| `REVERSE_HETERO` | `permute:40` | — | fails with libstdc++ only | D5 |
| `ROTATE_COPY_HETERO` | `write:76` | — | fails | D2 |
| `SET_DIFFERENCE` | `merge:55`, `merge:121` | all backends | fails | H1, D6; D1 for `merge:121` |
| `SET_INTERSECTION` | `merge:68`, `merge:134` | all backends | fails | H1, D6; D1 for `merge:134` |
| `SET_SYMMETRIC_DIFFERENCE` | `merge:82`, `merge:148` | all backends | fails | H1, D6; D1 for `merge:148` |
| `SET_UNION` | `merge:41`, `merge:107` | all backends | fails | H1, D6; D1 for `merge:107` |
| `SORT` | `permute:101`, `permute:233`, `permute:323`, `permute:389`, `permute:415` | OpenMP; TBB too for `permute:233` | fails only for `permute:389` | H2; H3 for `permute:233`; D6 for `permute:389` |
| `STABLE_PARTITION_HETERO` | `permute:222` | — | fails | D1 |
| `STABLE_SORT` | `permute:113`, `permute:245`, `permute:333`, `permute:402` | OpenMP; for `permute:245` TBB too, and with libstdc++ serial too | fails only for `permute:402` | H2; H3 and H4 for `permute:245`; D6 for `permute:402` |
| `UNIQUE_COPY_HETERO` | `write:182`, `write:380` | — | fails | D3 |
| `WRONG_RESULT_PARTIAL_SORT_COPY_PROJ1` | `std_ranges_partial_sort_copy.pass.cpp:78` | `par` and `par_unseq` differ from `seq` | differs from `seq` | T1 |

The `_REQUIRES_` gates are shown without the common prefix `_TEST_CPP20_RANGES_BROKEN_REQUIRES_`, the wrong-result
gate without `_TEST_CPP20_RANGES_BROKEN_`. A call site `<family>:<line>` is a line of
`conformance/std_ranges_algo_archetypes_<family>.pass.cpp`. The per-call results, with the backends and the standard
libraries apart, are in [Appendix A](#appendix-a-gated-calls).

**What the classification shows.**
- **The defects are few.** Twelve library classes and one test defect explain all 60 calls. 23 of the 32
  `_REQUIRES_` gates have a single class: D1 seven of them, D3 five, H3 four, D4 three, H6 two, D2 and D5 one each.
  The other nine (the four set algorithms, `sort`, `stable_sort`, `partial_sort`, `partial_sort_copy` and
  `inplace_merge`) combine two to four classes, so each of them needs more than one fix before it can be removed.
  The most frequent classes are D1 (in 12 gates), H3 (9), D6 (6), D3 and H1 (5 each).
- **One label is wrong.** `LEXICOGRAPHICAL_COMPARE` compiles on the host in every cell; only the device fails (D1).
  The host defect the gate was introduced for has been fixed. The gate should become `..._HETERO`, with the call
  split.
- **Unsuffixed gates also cover nine device calls that compile.** By the collapse rule, `SORT`, `STABLE_SORT` and
  `INPLACE_MERGE` have no suffix because one call of each fails on the device too (`permute:389`, `permute:402` and
  `permute:309`). Their nine other calls (`permute:101`, `:113`, `:172`, `:233`, `:245`, `:323`, `:333`, `:372`,
  `:415`) compile on the device, but their device halves are compiled out as well.
- **Some gaps exist with one backend or one standard library only.** H2 is the OpenMP backend: the host halves of
  the sort calls with a `const` comparator, the default comparator, a projected key or a plain range fail only
  with OpenMP, and compile with TBB and with the serial backend. D5 and H4 appear only with libstdc++. With g++,
  H6 starts with an extra error of its own.
- **One gate covers a test defect (T1).** `test_projections_consistency()` uses projections that violate the
  precondition of `partial_sort_copy`, so the behavior is undefined. The library behavior that the test reports (`par`,
  `par_unseq` and the device apply only `proj2`) is allowed by that precondition. Even `seq` gives different results
  with the two standard libraries: compared with `par`, the result of the Microsoft STL differs in 8201 of 16385
  elements, starting at index 8184, and the result of libstdc++ only in the last element. The test should use
  projections that satisfy the precondition, and the gate should be removed.

#### Retiring a Gate

Set the macro to `0`, then build and run the targets that use it, in the modes its suffix names (with several gates
open at once, open them one call at a time, see the method above). If they pass, the fix is in, and the macro and
its `#if` blocks are removed. Four gates have been retired this way
after library fixes so far:
- `_TEST_BROKEN_WRONG_RESULT_FIND_FIRST_OF_UNSEQ`, a former gate of the iterator-level test
  `alg.nonmodifying/find_first_of.pass.cpp`, and `_TEST_CPP20_RANGES_BROKEN_WRONG_RESULT_FIND_FIRST_OF_PROJ1_HOST`
  were removed after the `__simd_find_first_of()` fix;
- `_TEST_CPP20_RANGES_BROKEN_REQUIRES_FIND_FIRST_OF` was removed after the same fix, and its remaining device case
  was narrowed to `_TEST_CPP20_RANGES_BROKEN_REQUIRES_FIND_FIRST_OF_HETERO`;
- `_TEST_CPP20_RANGES_BROKEN_WRONG_RESULT_LEXICOGRAPHICAL_COMPARE_PROJ1_HOST` was removed after the fix of the
  vectorized `__brick_lexicographical_compare`.

### Building and Running

- **Discovery and labels.** The targets are ordinary `*.pass.cpp` files. `test/CMakeLists.txt` picks them up with
  `GLOB_RECURSE` and labels them `parallel_api;ranges;conformance`. Running `ctest -L ranges` or
  `ctest -L conformance` selects them.
- **Standard.** They need C++20 with `std::ranges` (`_ENABLE_STD_RANGES_TESTING`); otherwise they compile to a skip.
- **Device part.** It is compiled when `TEST_DPCPP_BACKEND_PRESENT` and uses USM shared allocations.
- **Public CI.** The names match the `std_ranges_.*` filter that public CI uses for the DPC++ backend on non-FPGA
  devices (`.github/scripts/run-linux-tests.sh`), so they run in those device cells as well. They build cleanly with
  the CI warning flags (`-Wall -Werror`, `/W4 /WX`); the compile-out idiom above exists for that.
- **Cost.** It is small, because every range has 1000 elements; the memory family reuses `medium_size`.

### Limitations

These are known, accepted limits of the current suite. Some of them are listed as [Open Questions](#open-questions).
- **Weak checks for sorting.** The input is already sorted ascending, so `sort`, `stable_sort`, `partial_sort`,
  `nth_element`, `inplace_merge` and `is_sorted` with an ascending comparator cannot tell a correct result from a
  no-op. These calls mainly check that the algorithm compiles and runs.
- **Incomplete output checks.** Some checks look only at a few elements or at the returned size: `remove` checks
  only the size of the removed tail, `merge` only the first values and the end iterator, `set_difference` an
  empty result.
- **The `_dc` twins are copyable.** On the device the suite does not check "non-copyable" or "move-only"
  (`nocopy_search_value_dc` is copyable despite its name).
- **Default predicates are homogeneous only.** Heterogeneous use of the default `ranges::equal_to` and
  `ranges::less` is not covered.
- **Plain-view coverage is a sample.** It covers `for_each`, `find_if`, `equal`, `copy` and `sort`.
- **Memory algorithms over `archetype_view`.** The memory family runs only `uninitialized_default_construct`,
  `uninitialized_value_construct` and `destroy` over it; the others run over `subrange` and `span`. The
  `uninitialized_copy` and `uninitialized_move` checks cannot detect a wrong order, because all inputs are equal.
- **Indeterminate values and unconstructed objects.** Like the existing memory tests, some checks read fields that
  no constructor wrote and run `destroy` over storage where no object was constructed. A fully conforming variant
  needs `std::byte` sentinels in the shared harness.
- **`lexicographical_compare` has no compile-level projection check.** It has only the data-level cases.

### Alternatives Considered

- **Concept checks only** (`static_assert(std::ranges::...)` on the algorithm objects). This is cheap, but it proves
  only that the declarations are constrained correctly. All the defects found so far were in the bodies.
- **Reusing the types from `test/support`.** They are designed for functional tests and provide too much; making
  them minimal would change the existing tests.
- **Fixing the library in the same PR.** The test PR would become hard to review and would mix many unrelated
  fixes. Gating keeps it test-only. The fixes go into separate PRs, each removing its gate.
- **Finer gate suffixes** (`_HOST_PARALLEL`, `_HOST_SIMD`, one gate per backend). They were tried and dropped: the
  host side is reported as a whole, and finer gates mostly add bookkeeping. Gates that depend on the backend or the
  standard library are a separate question, see [Open Questions](#open-questions).
- **Data-level projection checks only.** They depend on the generators and are easy to defeat. The compile-level
  check with `A` and `B` is exact and costs one call per algorithm.

## Open Questions

- **Gate lifecycle.** Should the macros stay hard-coded in `test_config.h`, or be configurable, for example to
  allow an "all gates off" build in CI that measures the remaining gaps? Should each macro carry a comment with the
  root cause and the affected `file:line`?
- **Labels after the classification.** Should `LEXICOGRAPHICAL_COMPARE` become `..._HETERO` (its host half compiles
  everywhere)? Should the collapse rule give way when an unsuffixed gate also compiles out device calls that compile:
  nine such calls are under `SORT`, `STABLE_SORT` and `INPLACE_MERGE`?
- **Backend- and library-specific gaps.** H2 exists only with the OpenMP backend, D5 and H4 only with libstdc++. A
  gate is defined to `1` everywhere, so these calls lose coverage on every backend and standard library. Should such
  gates depend on the configuration, for example `ONEDPL_USE_OPENMP_BACKEND` or `_GLIBCXX_RELEASE`?
- **Standard-library defects.** H4 is in libstdc++'s `std::ranges::stable_sort` and `std::ranges::inplace_merge`,
  which the host patterns call. Should oneDPL stop delegating there, or should the gap be reported upstream and the
  gate kept?
- **The `partial_sort_copy` projection test (T1).** Replace the projections of `test_projections_consistency()`
  with a pair that satisfies [partial.sort.copy]/4, and remove
  `_TEST_CPP20_RANGES_BROKEN_WRONG_RESULT_PARTIAL_SORT_COPY_PROJ1`?
- **Stronger checks.** Use shuffled or descending input for the sorting algorithms, check the full output of
  `remove`, `merge` and the set algorithms, and use distinct inputs for `uninitialized_copy` and
  `uninitialized_move`.
- **Device coverage of non-copyable types.** Can some device-copyable archetypes keep a deleted copy assignment, or
  is `sycl::is_device_copyable` specialization needed for that?
- **Wider coverage.** Heterogeneous default predicates, more `plain_archetype_view` calls, the remaining
  `oneapi::dpl::ranges` algorithms, and the compile-level projection check for `lexicographical_compare`.
- **Conforming memory tests.** Move the shared memory harness to `std::byte` sentinels and construct objects before
  `destroy`, in the existing tests as well.
- **Devices without USM shared allocations.** The conformance driver assumes them, while the projection helpers skip
  such devices.

## Appendix A: Gated Calls

One row per gated call, in the order of the families. "Call" is the name the call reports, "not gated" means that
this half of the call runs in the test today, and the classes are those of the
[classification](#classification-of-the-gated-calls). The host results of g++ 13 equal those of icpx with libstdc++
in every cell; where the two standard libraries differ, both are given.

| Call site | Call | Gate | Host policies | Device policy | Classes |
|---|---|---|---|---|---|
| `read:58` | find_last_if | `FIND_LAST_IF_HETERO` | not gated | fails | D4 |
| `read:73` | find_last_if_not | `FIND_LAST_IF_NOT_HETERO` | not gated | fails | D4 |
| `read:206` | find_last_if, non-const callable | `FIND_LAST_IF_HETERO` | not gated | fails | D4 |
| `read:221` | find_last_if_not, non-const callable | `FIND_LAST_IF_NOT_HETERO` | not gated | fails | D4 |
| `read:251` | is_partitioned, non-const callable | `IS_PARTITIONED_HETERO` | not gated | fails | D1 |
| `read:308` | is_heap, non-const comparator | `IS_HEAP_HETERO` | not gated | fails | D1 |
| `read:320` | is_heap_until, non-const comparator | `IS_HEAP_UNTIL_HETERO` | not gated | fails | D1 |
| `value:47` | find_last | `FIND_LAST_HETERO` | not gated | fails | D4 |
| `value:98` | find_last, noncopyable value | `FIND_LAST_HETERO` | not gated | fails | D4 |
| `value:156` | find_last, non-const projection | `FIND_LAST_HETERO` | not gated | fails | D4 |
| `cross:146` | find_first_of, non-const callable | `FIND_FIRST_OF_HETERO` | not gated | fails | D1 |
| `cross:177` | lexicographical_compare, non-const comparator | `LEXICOGRAPHICAL_COMPARE` | compiles | fails | D1 |
| `write:76` | rotate_copy | `ROTATE_COPY_HETERO` | not gated | fails | D2 |
| `write:150` | copy_if | `COPY_IF_HETERO` | not gated | fails | D3 |
| `write:166` | remove_copy_if | `REMOVE_COPY_IF_HETERO` | not gated | fails | D3 |
| `write:182` | unique_copy | `UNIQUE_COPY_HETERO` | not gated | fails | D3 |
| `write:202` | partition_copy | `PARTITION_COPY_HETERO` | not gated | fails | D3 |
| `write:241` | remove_copy | `REMOVE_COPY_HETERO` | not gated | fails | D3 |
| `write:258` | replace_copy_if | `REPLACE_COPY_IF_HOST` | all backends | not gated | H6 |
| `write:279` | replace_copy | `REPLACE_COPY_HOST` | all backends | not gated | H6 |
| `write:343` | copy_if, non-const predicate | `COPY_IF_HETERO` | not gated | fails | D3 |
| `write:362` | remove_copy_if, non-const predicate | `REMOVE_COPY_IF_HETERO` | not gated | fails | D3 |
| `write:380` | unique_copy, non-const relation | `UNIQUE_COPY_HETERO` | not gated | fails | D3 |
| `write:402` | partition_copy, non-const predicate | `PARTITION_COPY_HETERO` | not gated | fails | D3 |
| `write:444` | remove_copy, non-const projection | `REMOVE_COPY_HETERO` | not gated | fails | D3 |
| `write:462` | replace_copy_if, non-const predicate | `REPLACE_COPY_IF_HOST` | all backends | not gated | H6 |
| `write:484` | replace_copy, non-const projection | `REPLACE_COPY_HOST` | all backends | not gated | H6 |
| `permute:40` | reverse | `REVERSE_HETERO` | not gated | MSVC STL: compiles; libstdc++: fails | D5 |
| `permute:101` | sort | `SORT` | OpenMP | compiles | H2 |
| `permute:113` | stable_sort | `STABLE_SORT` | OpenMP | compiles | H2 |
| `permute:141` | partial_sort | `PARTIAL_SORT_HOST` | OpenMP | not gated | H2 |
| `permute:150` | partial_sort_copy | `PARTIAL_SORT_COPY` | all backends | fails | D2, H1, H2 |
| `permute:172` | inplace_merge | `INPLACE_MERGE` | all backends, through `seq` | compiles | H5 |
| `permute:208` | partition, non-const callable | `PARTITION_HETERO` | not gated | fails | D1 |
| `permute:222` | stable_partition, non-const callable | `STABLE_PARTITION_HETERO` | not gated | fails | D1 |
| `permute:233` | sort, non-const comparator | `SORT` | TBB, OpenMP | compiles | H2, H3 |
| `permute:245` | stable_sort, non-const comparator | `STABLE_SORT` | MSVC STL: TBB, OpenMP; libstdc++: all backends | compiles | H2, H3, H4 |
| `permute:267` | partial_sort, non-const comparator | `PARTIAL_SORT_HOST` | TBB, OpenMP | not gated | H2, H3 |
| `permute:276` | partial_sort_copy, non-const callables | `PARTIAL_SORT_COPY` | all backends | fails | D2, H1, H2, H3 |
| `permute:300` | nth_element, non-const comparator | `NTH_ELEMENT_HOST` | all backends | not gated | H3 |
| `permute:309` | inplace_merge, non-const comparator | `INPLACE_MERGE` | all backends | fails | D1, H3, H4, H5 |
| `permute:323` | sort, default comparator | `SORT` | OpenMP | compiles | H2 |
| `permute:333` | stable_sort, default comparator | `STABLE_SORT` | OpenMP | compiles | H2 |
| `permute:355` | partial_sort, default comparator | `PARTIAL_SORT_HOST` | OpenMP | not gated | H2 |
| `permute:372` | inplace_merge, default comparator | `INPLACE_MERGE` | all backends, through `seq` | compiles | H5 |
| `permute:389` | sort, projected key | `SORT` | OpenMP | fails | D6, H2 |
| `permute:402` | stable_sort, projected key | `STABLE_SORT` | OpenMP | fails | D6, H2 |
| `permute:415` | sort, plain range | `SORT` | OpenMP | compiles | H2 |
| `merge:41` | set_union | `SET_UNION` | all backends | fails | D6, H1 |
| `merge:55` | set_difference | `SET_DIFFERENCE` | all backends | fails | D6, H1 |
| `merge:68` | set_intersection | `SET_INTERSECTION` | all backends | fails | D6, H1 |
| `merge:82` | set_symmetric_difference | `SET_SYMMETRIC_DIFFERENCE` | all backends | fails | D6, H1 |
| `merge:107` | set_union, non-const comparator | `SET_UNION` | all backends | fails | D1, D6, H1 |
| `merge:121` | set_difference, non-const comparator | `SET_DIFFERENCE` | all backends | fails | D1, D6, H1 |
| `merge:134` | set_intersection, non-const comparator | `SET_INTERSECTION` | all backends | fails | D1, D6, H1 |
| `merge:148` | set_symmetric_difference, non-const comparator | `SET_SYMMETRIC_DIFFERENCE` | all backends | fails | D1, D6, H1 |
| `storable:55` | min, non-const comparator | `MIN_HOST` | all backends | not gated | H3 |
| `storable:71` | max, non-const comparator | `MAX_HOST` | all backends | not gated | H3 |
| `storable:89` | minmax, non-const comparator | `MINMAX_HOST` | all backends | not gated | H3 |
| `std_ranges_partial_sort_copy.pass.cpp:78` | `test_projections_consistency()` | `WRONG_RESULT_PARTIAL_SORT_COPY_PROJ1` | `par`, `par_unseq` differ from `seq` | differs from `seq` | T1 |
