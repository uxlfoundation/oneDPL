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
#include <algorithm> // std::find
#include <cstddef>   // std::size_t
#include <memory>    // std::unique_ptr
#include <tuple>
#include <utility>   // std::move, std::index_sequence
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
    using base::operator=; // inherit assignments

    const sycl::queue& queue() const { return this->__q; }

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
take_and_check(hetero::__device_storage<T>& storage, inspectable_holder<NScratch, ResultTypes...>& holder)
{
    void* const raw_ptr = storage.__usm_buf.get();
    const std::size_t count_before = holder.scratch_count();

    holder.__take(std::move(storage));

    EXPECT_TRUE(storage.__usm_buf == nullptr, "error in __take: the moved-from storage is not cleared");
    EXPECT_EQ(count_before + 1, holder.scratch_count(), "error in __take: scratch count change is not equal to 1");
    
    const auto& scratch_slot = holder.scratch_slot(count_before);
    EXPECT_EQ(raw_ptr, scratch_slot.__usm_ptr, // also holds for nullptr
              "error in __take: scratch slot does not hold the original USM pointer");
    EXPECT_EQ(raw_ptr == nullptr, scratch_slot.__sycl_buf.has_value(), 
              "error in __take: sycl::buffer was handled incorrectly");
}

template <std::size_t I, typename T, std::size_t NScratch, typename... ResultTypes>
void
take_and_check(hetero::__result_storage<T>& storage, inspectable_holder<NScratch, ResultTypes...>& holder)
{
    T* const raw_ptr = storage.__usm_buf.get();
    const sycl::usm::alloc kind = storage.__kind;
    const std::size_t count_before = holder.scratch_count();

    holder.template __take<I>(std::move(storage));
    
    EXPECT_TRUE(storage.__usm_buf == nullptr, "error in __take: the moved-from storage is not cleared");
    EXPECT_EQ(count_before, holder.scratch_count(), "error in __take: scratch count changed by result deposit");

    const auto& result_slot = holder.template result_slot<I>();
    EXPECT_EQ(raw_ptr, result_slot.__usm_ptr, // also holds for nullptr
              "error in __take: result slot does not hold the original USM pointer");
    EXPECT_EQ(kind == sycl::usm::alloc::unknown, result_slot.__sycl_buf.has_value(),
              "error in __take: sycl::buffer was handled incorrectly");
}

template <std::size_t I, typename T, std::size_t NScratch, typename... ResultTypes>
void
take_and_check(hetero::__combined_storage<T>& storage, inspectable_holder<NScratch, ResultTypes...>& holder)
{
    void* const scratch_raw = storage.__usm_buf.get();
    void* const result_raw  = storage.__result_buf.get();
    const sycl::usm::alloc kind = storage.__kind;
    const std::size_t count_before = holder.scratch_count();

    holder.template __take<I>(std::move(storage));

    EXPECT_TRUE(storage.__usm_buf == nullptr, "error in __take: the moved-from storage is not cleared");
    EXPECT_TRUE(storage.__result_buf == nullptr, "error in __take: the moved-from storage is not cleared");

    const auto& result_slot = holder.template result_slot<I>();
    EXPECT_EQ(kind == sycl::usm::alloc::unknown, result_slot.__sycl_buf.has_value(),
              "error in __take: sycl::buffer was handled incorrectly");
    if (kind == sycl::usm::alloc::host)
    {
        EXPECT_EQ(count_before + 1, holder.scratch_count(), "error in __take: scratch count change is not equal to 1");
        EXPECT_EQ(scratch_raw, holder.scratch_slot(count_before).__usm_ptr,
                  "error in __take: scratch slot does not hold the original USM pointer");
        EXPECT_EQ(result_raw, result_slot.__usm_ptr,
                  "error in __take: result slot does not hold the original USM pointer");
    }
    else
    {
        EXPECT_EQ(count_before, holder.scratch_count(), "error in __take: scratch count changed by combined deposit");
        EXPECT_EQ(scratch_raw, result_slot.__usm_ptr, // also holds for nullptr
                  "error in __take: result slot does not hold the original USM pointer");
    }
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

    template <std::size_t NScratch, typename... ResultTypes>
    void
    test_move(Test::inspectable_holder<NScratch, ResultTypes...>&& src)
    {
        const sycl::queue src_q     = src.queue();
        const std::size_t src_count = src.scratch_count();

        const auto check_moved_from = [](const auto& h)
        {
            EXPECT_EQ(0u, h.scratch_count(), "move: scratch count in moved-from holder must be 0");
            for (std::size_t s = 0; s < NScratch; ++s)
                EXPECT_TRUE(h.scratch_slot(s).__usm_ptr == nullptr,
                            "move: scratch slot in moved-from holder must be null");
            for (void* ptr : h.get_result_ptrs())
                EXPECT_TRUE(ptr == nullptr, "move: result slot in moved-from holder must be null");
        };
        const auto check_moved_into = [&](const auto& h)
        {
            EXPECT_EQ(src_count, h.scratch_count(), "move: scratch count in moved-into holder must match source");
            EXPECT_EQ(src_q, h.queue(), "move: queue in moved-into holder must match source");
        };

        // move construction
        Test::inspectable_holder<NScratch, ResultTypes...> dst{std::move(src)};
        check_moved_from(src);
        check_moved_into(dst);

        // move assignment
        Test::inspectable_holder<NScratch, ResultTypes...> dst2{sycl::queue{}};
        dst2 = std::move(dst);
        check_moved_from(dst);
        check_moved_into(dst2);
    }

    void test_scratch_deposits()
    {
        constexpr std::size_t NScratch = 3;
        Test::inspectable_holder<NScratch> holder{q};

        hetero::__device_storage<int> ds0(q, 314);
        hetero::__device_storage<char> ds1(q, 109);
        hetero::__device_storage<float> ds2(q, 83);

        std::array<void*, NScratch> raw_ptrs{ds0.__usm_buf.get(), ds1.__usm_buf.get(), ds2.__usm_buf.get()};

        Test::take_and_check(ds0, holder);
        Test::take_and_check(ds1, holder);
        Test::take_and_check(ds2, holder);

        EXPECT_EQ(NScratch, holder.scratch_count(), "scratch deposits: final scratch count is incorrect");
        if (kind != sycl::usm::alloc::unknown)
        {
            for (std::size_t s = 0; s < NScratch; ++s)
                EXPECT_EQ(raw_ptrs[s], holder.scratch_slot(s).__usm_ptr, "scratch deposits: a USM pointer lost or corrupt");
        }
        
        test_move(std::move(holder));
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

        Test::take_and_check<0>(rs0, holder);
        Test::take_and_check<1>(rs1, holder);
        Test::take_and_check<2>(rs2, holder);
        
        if (kind != sycl::usm::alloc::unknown)
        {
            std::array<void*, NResults> stored_ptrs = holder.get_result_ptrs();
            for (std::size_t s = 0; s < NResults; ++s)
                EXPECT_EQ(raw_ptrs[s], stored_ptrs[s], "result deposits: a USM pointer lost or corrupt");
        }

        test_move(std::move(holder));
    }

    void test_combined_deposits()
    {
        constexpr std::size_t NScratch = 3;
        Test::inspectable_holder<NScratch, int, float> holder{q};

        hetero::__combined_storage<int> cs0{q, 257, 2};
        hetero::__combined_storage<float> cs1{q, 99, 1};
        hetero::__device_storage<int> ds {q, 433};

        std::vector<void*> raw_ptrs{cs0.__usm_buf.get(), cs0.__result_buf.get(), ds.__usm_buf.get(),
                                    cs1.__usm_buf.get(), cs1.__result_buf.get()};

        Test::take_and_check<0>(cs0, holder);
        Test::take_and_check   (ds,  holder);
        Test::take_and_check<1>(cs1, holder);

        const std::size_t expected_scratch = /*ds*/1 + (kind == sycl::usm::alloc::host ? /*cs0&1*/2 : 0);
        EXPECT_EQ(expected_scratch, holder.scratch_count(), "combined deposits: final scratch count is incorrect");

        if (kind != sycl::usm::alloc::unknown)
        {
            auto check = [&](void* ptr)
            {
                auto it = std::find(raw_ptrs.begin(), raw_ptrs.end(), ptr);
                EXPECT_TRUE(it != raw_ptrs.end(), "combined deposits: unexpected pointer in a holder slot");
                if (it != raw_ptrs.end())
                    *it = nullptr;
            };

            for (void* ptr : holder.get_result_ptrs())
                check(ptr);

            for (std::size_t s = 0; s < NScratch; ++s)
            {
                if (s < expected_scratch)
                    check(holder.scratch_slot(s).__usm_ptr);
                else
                {
                    EXPECT_EQ(nullptr, holder.scratch_slot(s).__usm_ptr,
                              "combined deposits: unexpected pointer in a holder slot");
                }
            }

            for (void* ptr : raw_ptrs)
                EXPECT_TRUE(ptr == nullptr, "combined deposits: a USM pointer was lost");
        }

        test_move(std::move(holder));
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
        test_scratch_deposits();
        test_result_deposits();
        test_combined_deposits();

        if (kind == sycl::usm::alloc::unknown)
            return; // only limited testing for buffer-based storage
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
