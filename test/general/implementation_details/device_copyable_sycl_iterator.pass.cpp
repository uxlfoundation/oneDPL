// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"
using namespace TestUtils;

// The sweeping user specialization below uses a C++20 trailing requires-clause, which is required to make
// a partial specialization whose argument pattern is a bare template parameter well-formed.
#if TEST_DPCPP_BACKEND_PRESENT && _ONEDPL_CPP20_REQUIRES_CLAUSE_PRESENT

#    include "support/utils_device_copyable.h"
#    include "support/utils_sycl_defs.h"


// The following is ill-advised, but we need to guard against such actions a user may take
template <class T>
requires(true)
struct sycl::is_device_copyable<T> : std::true_type {};

#endif // TEST_DPCPP_BACKEND_PRESENT && _ONEDPL_CPP20_REQUIRES_CLAUSE_PRESENT

std::int32_t
main()
{
#if TEST_DPCPP_BACKEND_PRESENT && _ONEDPL_CPP20_REQUIRES_CLAUSE_PRESENT
    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::__internal::sycl_iterator<sycl::access_mode::read, int>>,
                  "sycl_iterator is device copyable but should never be");
#endif // TEST_DPCPP_BACKEND_PRESENT && _ONEDPL_CPP20_REQUIRES_CLAUSE_PRESENT
    return done(TEST_DPCPP_BACKEND_PRESENT && _ONEDPL_CPP20_REQUIRES_CLAUSE_PRESENT);
}
