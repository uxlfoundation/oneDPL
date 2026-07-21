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

template <int InitValue, typename InitT = int>
struct TestingAlgoritmExclusiveScan
{
    template <typename... TArgs>
    void
    call_onedpl(TArgs&&... args)
    {
        oneapi::dpl::exclusive_scan(std::forward<TArgs>(args)..., InitT(InitValue));
    }

    template <typename... TArgs>
    void
    call_serial(TArgs&&... args)
    {
        exclusive_scan_serial(std::forward<TArgs>(args)..., InitT(InitValue));
    }

    const char*
    getMsg(bool bInplace) const
    {
        return bInplace ? kMsgExclusiveScanInplace : kMsgExclusiveScanNormal;
    }
};

template <int InitValue, typename BinaryOp, typename InitT = int>
struct TestingAlgoritmExclusiveScanExt
{
    template <typename... TArgs>
    void
    call_onedpl(TArgs&&... args)
    {
        oneapi::dpl::exclusive_scan(std::forward<TArgs>(args)..., InitT(InitValue), BinaryOp());
    }

    template <typename... TArgs>
    void
    call_serial(TArgs&&... args)
    {
        exclusive_scan_serial(std::forward<TArgs>(args)..., InitT(InitValue), BinaryOp());
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
    test2buffers<alloc_type, test_scan_non_inplace<ValueType, TestingAlgoritmExclusiveScan<2, ValueType>>>();
    test2buffers<alloc_type,
                 test_scan_non_inplace<ValueType, TestingAlgoritmExclusiveScanExt<2, BinaryOperation, ValueType>>>();

    // Inplace
    test1buffer<alloc_type, test_scan_inplace<ValueType, TestingAlgoritmExclusiveScan<2, ValueType>>>();
    test1buffer<alloc_type,
                test_scan_inplace<ValueType, TestingAlgoritmExclusiveScanExt<2, BinaryOperation, ValueType>>>();
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

    // Run tests with a non-trivially-copyable but device-copyable value type
    using NonTrivialValueType = arith_device_copyable;
    using NonTrivialBinaryOperation = std::plus<NonTrivialValueType>;
    run_test<sycl::usm::alloc::shared, NonTrivialValueType, NonTrivialBinaryOperation>();
    run_test<sycl::usm::alloc::device, NonTrivialValueType, NonTrivialBinaryOperation>();

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
