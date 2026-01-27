// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test is a simple standalone SYCL test which is meant to prove that the SYCL installation is correct.
// If this test fails, it means that the SYCL environment has not be configured properly.

#include <iostream>

#define _SKIP_RETURN_CODE 77

#if ((defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)) &&                                          \
     (__has_include(<sycl/sycl.hpp>) || __has_include(<CL/sycl.hpp>))) &&                                             \
    (!defined(ONEDPL_USE_DPCPP_BACKEND) || ONEDPL_USE_DPCPP_BACKEND != 0)
#    define TEST_DPCPP_BACKEND_PRESENT 1
#else
#    define TEST_DPCPP_BACKEND_PRESENT 0
#endif

#if TEST_DPCPP_BACKEND_PRESENT

#    include <cstdlib>

#    if _MSC_VER && __INTEL_LLVM_COMPILER <= 20240000
// Algorithm is required as a workaround for a bug on windows before icx 2024.0.0 with sycl headers
// not including the appropriate headers for std::iter_swap
#        include <algorithm>
#    endif

#    if __has_include(<sycl/sycl.hpp>)
#        include <sycl/sycl.hpp>
#    else
#        include <CL/sycl.hpp>
#    endif

// Combine SYCL runtime library version
#    if defined(__LIBSYCL_MAJOR_VERSION) && defined(__LIBSYCL_MINOR_VERSION) && defined(__LIBSYCL_PATCH_VERSION)
#        define TEST_LIBSYCL_VERSION                                                                                   \
            (__LIBSYCL_MAJOR_VERSION * 10000 + __LIBSYCL_MINOR_VERSION * 100 + __LIBSYCL_PATCH_VERSION)
#    else
#        define TEST_LIBSYCL_VERSION 0
#    endif

inline auto default_selector =
#    if TEST_LIBSYCL_VERSION >= 60000
    sycl::default_selector_v;
#    else
    sycl::default_selector{};
#    endif

template <typename _Buf>
auto
__get_host_access(_Buf&& __buf)
{
#    if TEST_LIBSYCL_VERSION >= 60200
    return std::forward<_Buf>(__buf).get_host_access(sycl::read_only);
#    else
    return std::forward<_Buf>(__buf).template get_access<sycl::access::mode::read>();
#    endif
}

class canary_test_name;

int
test()
{
    const int count = 10;
    sycl::queue q(default_selector);
    sycl::buffer<int> buf(count);
    q.submit([&](sycl::handler& cgh) {
        sycl::accessor acc(buf, cgh, sycl::write_only);
        cgh.parallel_for<canary_test_name>(sycl::range</*dim=*/1>(count), [=](sycl::item</*dim=*/1> item_id) {
            auto idx = item_id.get_linear_id();
            acc[idx] = idx;
        });
    });
    auto host_acc = __get_host_access(buf);
    for (int i = 0; i < count; ++i)
    {
        if (host_acc[i] != i)
        {
            std::cout << "Failed\n";
            return 1;
        }
    }
    return 0;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
#    if _MSC_VER
    char* env_value = nullptr;
    size_t len;
    errno_t err = _dupenv_s(&env_value, &len, "_ONEDPL_SKIP_SYCL_CANARY_TEST");
    if (err)
    {
        std::cout << "Environment variable gather failed\n";
        return 1;
    }
#    else  // _MSC_VER
    const char* env_value = std::getenv("_ONEDPL_SKIP_SYCL_CANARY_TEST");
#    endif // _MSC_VER
    bool skip_sycl_canary_test = (env_value != nullptr);
    // This environment variable allows our main CI run to skip this test and not count it toward oneDPL's test
    // statistics, while still allowing non-ci test runs to have this as a environment health indicater.
    if (!skip_sycl_canary_test)
    {
        return test();
    }
#endif // TEST_DPCPP_BACKEND_PRESENT
    std::cout << "Skipped\n";
    return _SKIP_RETURN_CODE;
}
