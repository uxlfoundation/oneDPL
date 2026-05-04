// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"

#include "oneapi/dpl/execution"
#include "oneapi/dpl/numeric"
#include "oneapi/dpl/iterator"

#include "support/utils.h"
#include "support/scan_serial_impl.h"
#include "support/utils_scan.h"

using namespace TestUtils;

#if TEST_DPCPP_BACKEND_PRESENT
static const char kMsgExclusiveScanNormal [] = "Wrong effect from exclusive scan (non-inplace)";
static const char kMsgExclusiveScanInplace[] = "Wrong effect from exclusive scan (inplace)";

template <int InitValue>
struct TestingAlgoritmExclusiveScan
{
    template <typename... TArgs>
    void
    call_onedpl(TArgs&&... args)
    {
        oneapi::dpl::exclusive_scan(std::forward<TArgs>(args)..., InitValue);
    }

    template <typename... TArgs>
    void
    call_serial(TArgs&&... args)
    {
        exclusive_scan_serial(std::forward<TArgs>(args)..., InitValue);
    }

    const char*
    getMsg(bool bInplace) const
    {
        return bInplace ? kMsgExclusiveScanInplace : kMsgExclusiveScanNormal;
    }
};

template <int InitValue, typename BinaryOp>
struct TestingAlgoritmExclusiveScanExt
{
    template <typename... TArgs>
    void
    call_onedpl(TArgs&&... args)
    {
        oneapi::dpl::exclusive_scan(std::forward<TArgs>(args)..., InitValue, BinaryOp());
    }

    template <typename... TArgs>
    void
    call_serial(TArgs&&... args)
    {
        exclusive_scan_serial(std::forward<TArgs>(args)..., InitValue, BinaryOp());
    }

    const char*
    getMsg(bool bInplace) const
    {
        return bInplace ? kMsgExclusiveScanInplace : kMsgExclusiveScanNormal;
    }
};

template <sycl::usm::alloc alloc_type, typename ValueType, typename BinaryOperation>
void
run_test()
{
    // Non inplace
    test2buffers<alloc_type, test_scan_non_inplace<ValueType, TestingAlgoritmExclusiveScan<2>>>();
    test2buffers<alloc_type, test_scan_non_inplace<ValueType, TestingAlgoritmExclusiveScanExt<2, BinaryOperation>>>();

    // Inplace
    test1buffer<alloc_type, test_scan_inplace<ValueType, TestingAlgoritmExclusiveScan<2>>>();
    test1buffer<alloc_type, test_scan_inplace<ValueType, TestingAlgoritmExclusiveScanExt<2, BinaryOperation>>>();
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    using ValueType = int;
    using BinaryOperation = std::plus<ValueType>;

    // Run tests for USM shared memory
    run_test<sycl::usm::alloc::shared, ValueType, BinaryOperation>();
    // Run tests for USM device memory
    run_test<sycl::usm::alloc::device, ValueType, BinaryOperation>();

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
