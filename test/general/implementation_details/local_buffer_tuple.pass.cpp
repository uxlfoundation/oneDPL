// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <tuple>
#include <type_traits>

#include <oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h>

// test that __local_buffer maps a sycl::buffer element type to the type oneDPL uses internally, converting
// std::tuple to oneapi::dpl::__internal::tuple. The conversion must rebind the allocator too.
void
test_local_buffer_tuple_conversion()
{
    namespace __bknd = oneapi::dpl::__par_backend_hetero::__internal;

    using __internal_tuple = oneapi::dpl::__internal::tuple<int, double>;

    // A std::tuple buffer converts its element type to __internal::tuple.
    using __std_tuple_buffer = sycl::buffer<std::tuple<int, double>>;
    using __converted = typename __bknd::__local_buffer<__std_tuple_buffer>::type;

    // The element type is converted std::tuple -> __internal::tuple.
    static_assert(std::is_same_v<typename __converted::value_type, __internal_tuple>,
                  "__local_buffer must convert the element type of a std::tuple buffer to __internal::tuple");

    // The allocator must be rebound to the converted element type as well.
    static_assert(std::is_same_v<typename __converted::allocator_type, sycl::buffer<__internal_tuple>::allocator_type>,
                  "__local_buffer must rebind the allocator of a std::tuple buffer to __internal::tuple");
    static_assert(!std::is_same_v<typename __converted::allocator_type, __std_tuple_buffer::allocator_type>,
                  "__local_buffer must not leave the allocator bound to std::tuple");

    // A non-tuple buffer is passed through unchanged.
    using __int_buffer = sycl::buffer<int>;
    static_assert(std::is_same_v<typename __bknd::__local_buffer<__int_buffer>::type, __int_buffer>,
                  "__local_buffer must pass a non-tuple buffer through unchanged");
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    test_local_buffer_tuple_conversion();
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
