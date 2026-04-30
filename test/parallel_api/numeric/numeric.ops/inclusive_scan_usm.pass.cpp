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
#include "support/utils_invoke.h" // CLONE_TEST_POLICY_IDX
#include "support/scan_serial_impl.h"
#include "support/utils_scan.h"

using namespace TestUtils;

#if TEST_DPCPP_BACKEND_PRESENT
static const char kMsgInclusiveScanNormal [] = "Wrong effect from inclusive scan (non-inplace)";
static const char kMsgInclusiveScanInplace[] = "Wrong effect from inclusive scan (inplace)";

struct TestingAlgoritmInclusiveScan
{
    template <typename... TArgs>
    void
    call_onedpl(TArgs&&... args)
    {
        oneapi::dpl::inclusive_scan(std::forward<TArgs>(args)...);
    }

    template <typename... TArgs>
    void
    call_serial(TArgs&&... args)
    {
        inclusive_scan_serial(std::forward<TArgs>(args)...);
    }

    const char*
    getMsg(bool bInplace) const
    {
        return bInplace ? kMsgInclusiveScanInplace : kMsgInclusiveScanNormal;
    }
};

template <typename BinaryOp, int InitValue>
struct TestingAlgoritmInclusiveScanExt
{
    template <typename... TArgs>
    void
    call_onedpl(TArgs&&... args)
    {
        oneapi::dpl::inclusive_scan(std::forward<TArgs>(args)..., BinaryOp(), InitValue);
    }

    template <typename... TArgs>
    void
    call_serial(TArgs&&... args)
    {
        inclusive_scan_serial(std::forward<TArgs>(args)..., BinaryOp(), InitValue);
    }

    const char*
    getMsg(bool bInplace) const
    {
        return bInplace ? kMsgInclusiveScanInplace : kMsgInclusiveScanNormal;
    }
};

template <sycl::usm::alloc alloc_type, typename ValueType, typename BinaryOperation>
void
run_test()
{
    // Non inplace
    test2buffers<alloc_type, test_scan_non_inplace<ValueType, TestingAlgoritmInclusiveScan> >();
    test2buffers<alloc_type, test_scan_non_inplace<ValueType, TestingAlgoritmInclusiveScanExt<BinaryOperation, 2 > > >();

    // Inplace
    test1buffer<alloc_type, test_scan_inplace<ValueType, TestingAlgoritmInclusiveScan>>();
    test1buffer<alloc_type, test_scan_inplace<ValueType, TestingAlgoritmInclusiveScanExt<BinaryOperation, 2 > > >();
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
