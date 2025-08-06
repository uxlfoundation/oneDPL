# Indirectly Device Accessible Trait and Customization Point for User-Defined Types

## Introduction
Iterators and iterator-like types may or may not refer to content accessible within a
[SYCL](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html) kernel on a device. The term
*indirectly device accessible* refers to a type that represents content accessible on a device. An indirectly device
accessible iterator is a type that can also be dereferenced within a SYCL kernel.

When passed to oneDPL algorithms with a device execution policy, indirectly device accessible types minimize
data movement and behave equivalently to using the type directly within a SYCL kernel.

oneDPL provides tools that allow users to customize their own user-defined iterators to be indirectly device accessible,
making them performant with oneDPL algorithms when using a device execution policy. For example, ``device_iterator`` and
``device_pointer`` [found in SYCLomatic helper headers](https://github.com/oneapi-src/SYCLomatic/blob/7fc89aff9e3aeeb8794b0f8baa33f5a65496868a/clang/runtime/dpct-rt/include/dpct/dpl_extras/memory.h)
currently use the unspecified alias ``using is_passed_directly = std::true_type;`` to indicate to oneDPL that the data
is already on the device. Similarly, [Kokkos uses the same alias](https://github.com/kokkos/kokkos/pull/7502) to
provide this functionality to a ``RandomAccessIterator`` which may be strided but is known to be on the device. This
feature allows more flexibility to express this trait either in the class implementation or alongside it, without
leaving the user's namespace.

### Changes from proposed RFC
The original RFC as proposed focused more on the implementation detail "is passed directly", where during discussion
of the specification, we shifted the semantic meaning of the public trait to only describe whether the type refers
to data which is accessible on the device. This separates its meaning from other requirements for direct usage within
a SYCL kernel like random access iterator or SYCL device copyable. The name indirectly device accessible was selected
to represent this trait.

## Design and Implementation

### New Public Trait and Customization Point

We have added the trait ``template <typename T> oneapi::dpl::is_indirectly_device_accessible[_v]``, which has base
characteristics of ``std::true_type`` when ``T`` is *indirectly device accessible*. There is also an argument-dependent
lookup (ADL)-based customization point function, ``is_onedpl_indirectly_device_accessible(T)``, which can be used to
define types as *indirectly device accessible*. For more details, see the
[oneDPL specification](https://github.com/uxlfoundation/oneAPI-spec/blob/main/source/elements/oneDPL/source/parallel_api/iterators.rst).

### Indirectly Device Accessible for oneDPL Inputs

The following table summarizes oneDPL input types and whether they are Indirectly Device Accessible:

| Input Type                                 | Indirectly Device Accessible                |
|--------------------------------------------|---------------------------------------------|
| counting_iterator                          | Yes                                         |
| discard_iterator                           | Yes                                         |
| zip_iterator                               | If all source iterators are indirectly device accessible |
| transform_iterator                         | If the source iterator is indirectly device accessible |
| permutation_iterator                       | If both source iterator and index map are indirectly device accessible |
| Return of oneapi::dpl::begin(), end()      | Yes                                         |
| USM pointers                               | Yes                                         |
| std::vector::iterator with a host allocator| No                                          |

The following are extensions within our implementation which are not a part of the oneDPL specification. Please read
the notes below for more information, and recommendations.

| Extension Input Type                       | Indirectly Device Accessible                |
|--------------------------------------------|---------------------------------------------|
| std::reverse_iterator                      | If the source iterator is indirectly device accessible |
| std::vector::iterator with a USM allocator | If the allocator is "known" in the vector iterator type  |
| Iterators using is_passed_directly = std::true_type | Yes                                |

oneDPL supports `std::reverse_iterator` for source iterators that are indirectly device accessible, as much as possible.
Users must adhere to the standard library specification when using `std::reverse_iterator`. The return values of
`oneapi::dpl::begin()` and `oneapi::dpl::end()` are not iterators, so they are not eligible for use within a
`std::reverse_iterator`. oneDPL makes no guarantee that `std::reverse_iterator` is sycl device-copyable, this is
dependent on the standard library implementation.

oneDPL attempts to determine if the allocator can be identified as a USM allocator within the type of the
`std::vector::iterator`. If so, it treats the iterator as indirectly device accessible. This relies on implementation
details of the standard library, and it is not always possible. For this reason, we recommend using `data()` on vectors
allocated with a USM allocator to obtain a USM pointer. USM pointers should work regardless of the standard library
implementation of `std::vector`.

The `is_passed_directly` alias as `std::true_type` provides legacy support to some helpers within SYCLomatic. This may
be removed in the future without deprecation, once there are no longer known features utilizing this unspecified
feature. These aliases should be replaced with overloads of `is_onedpl_indirectly_device_accessible()` to guarantee
continued functionality.


### Implementation Details

Internally, a trait `template <typename _Iter> oneapi::dpl::__ranges::__is_passed_directly_device_ready`
is defined as the conjunction of `oneapi::dpl::is_indirectly_device_accessible<_Iter>` and
`sycl::is_device_copyable<_Iter>`. Input types that are *passed directly device ready* are passed to
SYCL kernels without any additional processing for direct usage.

A static assertion is used to prevent incorrect usage of the ADL-based customization point
`is_onedpl_indirectly_device_accessible()` by returning a type which is not a `std::bool_constant`. This is meant to
alert users clearly of incorrect usage of the customization point.

All hidden friend functions for the `is_onedpl_indirectly_device_accessible()` customization point provide a minimal
body within oneDPL, as do our examples in the specification and documentation. This is because GCC 15.1 provides a
warning for hidden friend functions that are body-less, as such usage matches a common anti-pattern where a
non-template friend is declared but not defined within the class definition, and then cannot be defined externally
to the class with a matching function signature. There is technically no problem with a body-less hidden friend
function for our purposes, as we only need the function declaration for evaluating the return type in a `declval`
scope. However, it is better to avoid this warning and the appearance of any problems.

### Testing
`test/parallel_api/iterator/indirectly_device_accessible.pass.cpp` provides testing for a variety of types, and
recursions upon types and the value of their `is_indirectly_device_accessible` trait both in the negative and positive
result.

It would be good to extend `test/parallel_api/iterator/input_data_sweep.h` tests to include a custom iterator
adaptor which tests actual usage of a custom type within oneDPL using the trait in different scenarios. However, with
the tests of oneDPL's provided iterators, we do cover the functionality of this system because those types use the ADL
customization point to define themselves indirectly device accessible.

## Alternatives Considered 
Note: these are carried forward from the previous RFC, where the framing was more around the implementation detail of
whether a type "is passed directly" to a sycl kernel. We have since reframed that for the public API within the
specification. 

### Public Trait Struct Explicit Specialization
oneDPL could make public our internal structure `oneapi::dpl::__ranges::is_passed_directly` as
`oneapi::dpl::is_passed_directly` for users to specialize to define rules for their types. This would be a similar
mechanism to `sycl::is_device_copyable`. The implementation details of this option should avoid some complexity required
to properly implement the customization point.

However, as we have learned from experience within oneDPL, explicit specialization of a structure in another library's
namespace creates maintenance problems. It either requires lots of closing of nested namespaces, opening of the external
library's namespace for the specialization, or it requires separating these specializations to a separate location
removed from the types they are specializing for. oneDPL has chosen to use the latter, which can be seen in
`include/oneapi/dpl/pstl/hetero/dpcpp/sycl_traits.h`. This has led to several errors where changes to structures should
have included changes to sycl_traits but did not, and needed to be fixed later.

In an effort to avoid this same issue for our users, we propose a similar method but instead with a constexpr
customization point, allowing the user to override that customization point within their own namespace as a free
function.

### Require Specifically Named Typedef / Using in User's Type
oneDPL could make official our requirements for users' types to include a typedef or using statement to define if the
type is "passed directly" like `using is_passed_directly = std::true_type;`, where the absence of this would be
equivalent to a `std::false_type`.

However, this clutters the user type definitions with specifics of oneDPL. It also may not be as clear what this
signifies for maintenance of user code without appropriate comments describing the details of oneDPL and SYCL. Users
have expressed that this is undesirable.

### Wrapper Class
oneDPL could provide some wrapper iterator `direct_iterator` which wraps an arbitrary base iterator and marks it as
"passed directly". `direct_iterator` could utilize either of the above alternatives to accomplish this and signal that
the iterator should be "passed directly". It would need to pass through all operations to the wrapped base iterator and
make sure no overhead is added in its usage.

There is some complexity in adding such a wrapper iterator, and it would need to be considered carefully to make sure no
problems would be introduced. This wrapper class may obfuscate users' types and make them more unwieldy to use. It is
also less expressive than the other options in that it only has the ability to unilaterally mark a type as "passed
directly". There is no logic that can be used to express some iterator type which may be conditionally "passed
directly", other than to have logic to conditionally apply the wrapper in the first place. This option seems less clear
and has more opportunity to cause problems.


