// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Compile-time test for KT radix sort eligibility traits. This verifies that
// the shape detection and view compatibility checks correctly identify eligible
// and ineligible cases at compile time.

#include "support/test_config.h"
#include "support/utils.h"

#include <iostream>
#include <cstdint>

#if !TEST_DPCPP_BACKEND_PRESENT
int main() { return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT); }
#else

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(iterator)

#include "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_radix_sort_kt.h"
#include "oneapi/dpl/pstl/hetero/dpcpp/utils_ranges_sycl.h"
#include "oneapi/dpl/pstl/utils_ranges.h"

#if _ONEDPL_CPP20_RANGES_PRESENT
#include <span>
#endif

// Only test when the macro is enabled
#if !_ONEDPL_ENABLE_KT_RADIX_SORT_IN_SORT
int main()
{
    std::cout << "KT radix sort integration is disabled (ONEDPL_ENABLE_KT_RADIX_SORT_IN_SORT=0)" << std::endl;
    return TestUtils::done(/*is_done=*/false);
}
#else

namespace kr = oneapi::dpl::__par_backend_hetero::__kt_radix;
namespace ranges = oneapi::dpl::__ranges;

// Test view compatibility traits
static_assert(kr::__is_kt_radix_compatible_view<ranges::guard_view<std::uint32_t*>>,
              "guard_view over pointer should be compatible");

static_assert(kr::__is_kt_radix_compatible_view<
    ranges::all_view<std::uint32_t, sycl::access::mode::read_write, false,
                    __dpl_sycl::__target::device, sycl::access::placeholder::false_t>>,
              "all_view (buffer accessor) should be compatible");

static_assert(!kr::__is_kt_radix_compatible_view<
    ranges::permutation_view_simple<ranges::all_view<std::uint32_t, sycl::access::mode::read_write, false,
                    __dpl_sycl::__target::device, sycl::access::placeholder::false_t>,
                    ranges::all_view<std::uint32_t, sycl::access::mode::read_write, false,
                    __dpl_sycl::__target::device, sycl::access::placeholder::false_t>>>,
              "permutation_view_simple should not be compatible");

static_assert(!kr::__is_kt_radix_compatible_view<
    ranges::transform_view_simple<ranges::guard_view<std::uint32_t*>, oneapi::dpl::identity>>,
              "transform_view_simple should not be compatible");

#if _ONEDPL_CPP20_RANGES_PRESENT
// Contiguous range (std::span) should be compatible via the C++20 path
static_assert(kr::__is_kt_radix_compatible_view<std::span<std::uint32_t>>,
              "std::span should be compatible");
#endif

// Test shape detection

// Keys-only with USM pointer
static_assert(kr::__kt_radix_sort_shape<ranges::guard_view<std::uint32_t*>, oneapi::dpl::identity> ==
              kr::__kt_sort_shape::__keys_only,
              "guard_view with identity should be keys-only");

// Keys-only with buffer
static_assert(kr::__kt_radix_sort_shape<
    ranges::all_view<std::uint32_t, sycl::access::mode::read_write, false,
                    __dpl_sycl::__target::device, sycl::access::placeholder::false_t>,
    oneapi::dpl::identity> == kr::__kt_sort_shape::__keys_only,
              "all_view with identity should be keys-only");

// By-key with zip_view of USM pointers
using zip_usm = ranges::zip_view<ranges::guard_view<std::uint32_t*>, ranges::guard_view<std::uint64_t*>>;
static_assert(kr::__kt_radix_sort_shape<zip_usm, oneapi::dpl::__internal::__pattern_sort_by_key_fn> ==
              kr::__kt_sort_shape::__by_key,
              "zip_view with __pattern_sort_by_key_fn should be by-key");

// By-key with zip_view of buffers
using all_view_u32 = ranges::all_view<std::uint32_t, sycl::access::mode::read_write, false,
                    __dpl_sycl::__target::device, sycl::access::placeholder::false_t>;
using all_view_u64 = ranges::all_view<std::uint64_t, sycl::access::mode::read_write, false,
                    __dpl_sycl::__target::device, sycl::access::placeholder::false_t>;
using zip_buffer = ranges::zip_view<all_view_u32, all_view_u64>;
static_assert(kr::__kt_radix_sort_shape<zip_buffer, oneapi::dpl::__internal::__pattern_sort_by_key_fn> ==
              kr::__kt_sort_shape::__by_key,
              "zip_view of buffers with __pattern_sort_by_key_fn should be by-key");

#if _ONEDPL_CPP20_RANGES_PRESENT
// Keys-only with std::span (ranges::sort path)
static_assert(kr::__kt_radix_sort_shape<std::span<std::uint32_t>, oneapi::dpl::identity> ==
              kr::__kt_sort_shape::__keys_only,
              "std::span with identity should be keys-only");
#endif

// Negative cases

// permutation_view should be __none
static_assert(kr::__kt_radix_sort_shape<
    ranges::permutation_view_simple<all_view_u32, all_view_u32>,
    oneapi::dpl::identity> == kr::__kt_sort_shape::__none,
              "permutation_view_simple should be __none");

// zip_view with identity (not __pattern_sort_by_key_fn) should be __none
static_assert(kr::__kt_radix_sort_shape<zip_usm, oneapi::dpl::identity> == kr::__kt_sort_shape::__none,
              "zip_view with identity projection should be __none");

// By-key with non-contiguous values side should be __none
using zip_mixed = ranges::zip_view<
    ranges::guard_view<std::uint32_t*>,
    ranges::permutation_view_simple<all_view_u32, all_view_u32>>;
static_assert(kr::__kt_radix_sort_shape<zip_mixed, oneapi::dpl::__internal::__pattern_sort_by_key_fn> ==
              kr::__kt_sort_shape::__none,
              "zip_view with non-contiguous values should be __none");

// The assertions above use hand-written view types. Those confirm the traits behave as designed,
// but not that the sort plumbing actually *produces* those types — if __get_sycl_range's result
// ever changes shape, the traits would silently start rejecting everything and every sort would
// quietly fall back to the legacy path with no test failing. So repeat the checks over the view
// types derived from the real pipeline.

using keep_t = decltype(ranges::__get_sycl_range<oneapi::dpl::__par_backend_hetero::access_mode::read_write>());

template <typename Iterator>
using derived_view_t =
    std::decay_t<decltype(std::declval<keep_t&>()(std::declval<Iterator>(), std::declval<Iterator>()).all_view())>;

template <typename Iterator, typename Proj>
inline constexpr auto derived_shape = kr::__kt_radix_sort_shape<derived_view_t<Iterator>, Proj>;

using by_key_proj = oneapi::dpl::__internal::__pattern_sort_by_key_fn;
using buf_u32_it = decltype(oneapi::dpl::begin(std::declval<sycl::buffer<std::uint32_t>&>()));
using buf_u64_it = decltype(oneapi::dpl::begin(std::declval<sycl::buffer<std::uint64_t>&>()));

struct doubler
{
    int
    operator()(int i) const
    {
        return 2 * i;
    }
};

static_assert(derived_shape<std::uint32_t*, oneapi::dpl::identity> == kr::__kt_sort_shape::__keys_only,
              "sort over a USM pointer must reach KT");
static_assert(derived_shape<buf_u32_it, oneapi::dpl::identity> == kr::__kt_sort_shape::__keys_only,
              "sort over a sycl::buffer must reach KT");
static_assert(derived_shape<decltype(oneapi::dpl::make_zip_iterator(std::declval<std::uint32_t*>(),
                                                                    std::declval<std::uint64_t*>())),
                            by_key_proj> == kr::__kt_sort_shape::__by_key,
              "sort_by_key over USM pointers must reach KT");
static_assert(derived_shape<decltype(oneapi::dpl::make_zip_iterator(std::declval<buf_u32_it>(),
                                                                    std::declval<buf_u64_it>())),
                            by_key_proj> == kr::__kt_sort_shape::__by_key,
              "sort_by_key over sycl::buffers must reach KT");

static_assert(derived_shape<decltype(oneapi::dpl::make_permutation_iterator(std::declval<std::uint32_t*>(),
                                                                            doubler{})),
                            oneapi::dpl::identity> == kr::__kt_sort_shape::__none,
              "permutation_iterator must fall back to the legacy path");
static_assert(derived_shape<std::reverse_iterator<std::uint32_t*>, oneapi::dpl::identity> ==
                  kr::__kt_sort_shape::__none,
              "reverse_iterator must fall back to the legacy path");

int main()
{
    std::cout << "All compile-time trait assertions passed" << std::endl;
    return TestUtils::done();
}

#endif // _ONEDPL_ENABLE_KT_RADIX_SORT_IN_SORT
#endif // TEST_DPCPP_BACKEND_PRESENT
