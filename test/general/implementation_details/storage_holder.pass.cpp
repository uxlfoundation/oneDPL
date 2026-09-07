// -*- C++ -*-
//===------------------------------------------------------===//
//
// Copyright (C) 2025 UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#include <oneapi/dpl/pstl/hetero/dpcpp/utils_storage_sycl.h>

#include <array>
#include <cstddef>  // std::size_t
#include <memory>   // std::unique_ptr
#include <tuple>
#include <utility>  // std::move, std::index_sequence
#include <vector>

namespace hetero   = oneapi::dpl::__par_backend_hetero;
namespace internal = oneapi::dpl::__par_backend_hetero::__internal;

namespace Test
{

template <std::size_t NScratch, typename... ResultTypes>
struct inspectable_holder : public hetero::__storage_holder<NScratch, ResultTypes...>
{
    using base = hetero::__storage_holder<NScratch, ResultTypes...>;
    using base::base; // inherit constructors

    static constexpr std::size_t result_count() { return sizeof...(ResultTypes); }
    auto scratch_count() const { return this->__scratch_count; }
    
    auto scratch_slot(std::size_t i) const { return this->__scratch_slots[i]; }
    template <std::size_t I>
    auto result_slot() const { return std::get<I>(this->__result_slots); }
    
    auto /*std::array*/ get_result_ptrs() const
    {
        return std::apply([](const auto&... slot){ return std::array<void*, result_count()>{slot.__usm_ptr...}; },
                          this->__result_slots);
    }
};

// Test helpers
template <typename T, std::size_t NScratch, typename... ResultTypes>
void
take_scratch_and_check(hetero::__device_storage<T>& storage,
                       inspectable_holder<NScratch, ResultTypes...>& holder)
{
    void* const raw_ptr = storage.__usm_buf.get();
    const std::size_t count_before = holder.scratch_count();

    holder.__take(std::move(storage));

    EXPECT_EQ(count_before + 1, holder.scratch_count(), "error in __take: scratch count change is not equal to 1");
    EXPECT_EQ(raw_ptr, holder.scratch_slot(count_before).__usm_ptr, // also holds for nullptr
              "error in __take: holder slot does not hold the original USM pointer");
    EXPECT_TRUE(storage.__usm_buf == nullptr, "error in __take: the moved-from storage is not cleared");
}

template <std::size_t I, typename T, std::size_t NScratch, typename... ResultTypes>
void
take_result_and_check(hetero::__result_storage<T>& storage,
                      inspectable_holder<NScratch, ResultTypes...>& holder)
{
    T* const raw_ptr = storage.__usm_buf.get();
    const std::size_t count_before = holder.scratch_count();

    holder.template __take<I>(std::move(storage));

    EXPECT_EQ(count_before, holder.scratch_count(), "error in __take: scratch count changed by result deposit");
    EXPECT_EQ(raw_ptr, holder.template result_slot<I>().__usm_ptr, // also holds for nullptr
              "error in __take: holder slot does not hold the original USM pointer");
    EXPECT_TRUE(storage.__usm_buf == nullptr, "error in __take: the moved-from storage is not cleared");
}

} // namespace Test

// Test struct
struct StorageHolderTest
{
    sycl::queue q;
    sycl::usm::alloc kind;
    
    StorageHolderTest(sycl::queue queue) : q(queue)
    {
        // determine which USM type will be used for result storage
        sycl::device dvc = q.get_device();
        if (dvc.has(sycl::aspect::usm_host_allocations))
            kind = sycl::usm::alloc::host;
        else if (dvc.has(sycl::aspect::usm_device_allocations))
            kind = sycl::usm::alloc::device;
        else
            kind = sycl::usm::alloc::unknown;
    }

    void test_scratch_deposits()
    {
        constexpr std::size_t NScratch = 3;
        Test::inspectable_holder<NScratch> holder{q};

        hetero::__device_storage<int> ds0(q, 314);
        hetero::__device_storage<char> ds1(q, 109);
        hetero::__device_storage<float> ds2(q, 83);

        std::array<void*, NScratch> raw_ptrs{ds0.__usm_buf.get(), ds1.__usm_buf.get(), ds2.__usm_buf.get()};

        Test::take_scratch_and_check(ds0, holder);
        Test::take_scratch_and_check(ds1, holder);
        Test::take_scratch_and_check(ds2, holder);

        EXPECT_EQ(NScratch, holder.scratch_count(), "scratch deposits: final count differs from the number of deposits");
        for (std::size_t s = 0; s < NScratch; ++s)
            EXPECT_EQ(raw_ptrs[s], holder.scratch_slot(s).__usm_ptr, "scratch deposits: a USM pointer lost or corrupt");
    }

    void test_result_deposits()
    {
        using TupleT = std::tuple<int, long>;
        Test::inspectable_holder<0, int, float, TupleT> holder{q};
        constexpr std::size_t NResults = holder.result_count();

        hetero::__result_storage<int> rs0(q, 217);
        hetero::__result_storage<float> rs1(q, 42);
        hetero::__result_storage<TupleT> rs2(q, 193);

        std::array<void*, NResults> raw_ptrs{rs0.__usm_buf.get(), rs1.__usm_buf.get(), rs2.__usm_buf.get()};

        Test::take_result_and_check<0>(rs0, holder);
        Test::take_result_and_check<1>(rs1, holder);
        Test::take_result_and_check<2>(rs2, holder);
        
        std::array<void*, NResults> stored_ptrs = holder.get_result_ptrs();
        for (std::size_t s = 0; s < NResults; ++s)
            EXPECT_EQ(raw_ptrs[s], stored_ptrs[s], "result deposits: a USM pointer lost or corrupt");
    }

    // Edge case: NScratch == 0, empty ResultTypes
    void test_empty_holder()
    {
        hetero::__storage_holder<0> src{q};
        hetero::__storage_holder<0> dst{std::move(src)};
        // Both must destruct cleanly — no slots, no counts to check
    }

    // Run all tests
    void run()
    {
        test_empty_holder();

        if (kind == sycl::usm::alloc::unknown)
            return; // only limited testing for buffer-based storage

        test_scratch_deposits();
        test_result_deposits();
    }
};

#endif // TEST_DPCPP_BACKEND_PRESENT

int main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    StorageHolderTest test{TestUtils::get_test_queue()};
    test.run();
#endif
    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
