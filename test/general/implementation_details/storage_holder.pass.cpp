// -*- C++ -*-
//===------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#include <oneapi/dpl/pstl/hetero/dpcpp/utils_storage_sycl.h>

#include <array>
#include <algorithm> // std::find, std::fill
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
    template <std::size_t I>
    auto& result_slot_ref() { return std::get<I>(this->__result_slots); }

    auto /*std::array*/ get_result_ptrs() const
    {
        return std::apply([](const auto&... slot){ return std::array<void*, result_count()>{slot.__usm_ptr...}; },
                          this->__result_slots);
    }
};

// Test helpers
template <typename T, std::size_t NScratch, typename... ResultTypes>
void
store_and_check(hetero::__device_storage<T>& storage, inspectable_holder<NScratch, ResultTypes...>& holder)
{
    void* const raw_ptr = storage.__usm_buf.get();
    const std::size_t count_before = holder.scratch_count();

    holder.__store_scratch(std::move(storage));

    EXPECT_TRUE(storage.__usm_buf == nullptr, "error in __store_scratch: the moved-from storage is not cleared");
    EXPECT_EQ(count_before + 1, holder.scratch_count(), "error in __store_scratch: scratch count change is not equal to 1");
    
    const auto& scratch_slot = holder.scratch_slot(count_before);
    EXPECT_EQ(raw_ptr, scratch_slot.__usm_ptr, // also holds for nullptr
              "error in __store_scratch: scratch slot does not hold the original USM pointer");
    EXPECT_EQ(raw_ptr == nullptr, scratch_slot.__sycl_buf.has_value(), 
              "error in __store_scratch: sycl::buffer was handled incorrectly");
}

template <std::size_t I, typename T, std::size_t NScratch, typename... ResultTypes>
void
store_and_check(hetero::__result_storage<T>& storage, inspectable_holder<NScratch, ResultTypes...>& holder)
{
    T* const raw_ptr = storage.__usm_buf.get();
    const sycl::usm::alloc kind = storage.__kind;
    const std::size_t count_before = holder.scratch_count();

    holder.template __store<I>(std::move(storage));
    
    EXPECT_TRUE(storage.__usm_buf == nullptr, "error in __store: the moved-from storage is not cleared");
    EXPECT_EQ(count_before, holder.scratch_count(), "error in __store: scratch count changed by result deposit");

    const auto& result_slot = holder.template result_slot<I>();
    EXPECT_EQ(raw_ptr, result_slot.__usm_ptr, // also holds for nullptr
              "error in __store: result slot does not hold the original USM pointer");
    EXPECT_EQ(kind == sycl::usm::alloc::unknown, result_slot.__sycl_buf.has_value(),
              "error in __store: sycl::buffer was handled incorrectly");
}

template <std::size_t I, typename T, std::size_t NScratch, typename... ResultTypes>
void
store_and_check(hetero::__combined_storage<T>& storage, inspectable_holder<NScratch, ResultTypes...>& holder)
{
    void* const scratch_raw = storage.__usm_buf.get();
    void* const result_raw  = storage.__result_buf.get();
    const sycl::usm::alloc kind = storage.__kind;
    const std::size_t count_before = holder.scratch_count();

    holder.template __store<I>(std::move(storage));

    EXPECT_TRUE(storage.__usm_buf == nullptr, "error in __store: the moved-from storage is not cleared");
    EXPECT_TRUE(storage.__result_buf == nullptr, "error in __store: the moved-from storage is not cleared");

    const auto& result_slot = holder.template result_slot<I>();
    EXPECT_EQ(kind == sycl::usm::alloc::unknown, result_slot.__sycl_buf.has_value(),
              "error in __store: sycl::buffer was handled incorrectly");
    if (kind == sycl::usm::alloc::host)
    {
        EXPECT_EQ(count_before + 1, holder.scratch_count(), "error in __store: scratch count change is not equal to 1");
        EXPECT_EQ(scratch_raw, holder.scratch_slot(count_before).__usm_ptr,
                  "error in __store: scratch slot does not hold the original USM pointer");
        EXPECT_EQ(result_raw, result_slot.__usm_ptr,
                  "error in __store: result slot does not hold the original USM pointer");
    }
    else
    {
        EXPECT_EQ(count_before, holder.scratch_count(), "error in __store: scratch count changed by combined deposit");
        EXPECT_EQ(scratch_raw, result_slot.__usm_ptr, // also holds for nullptr
                  "error in __store: result slot does not hold the original USM pointer");
    }
}

template <typename T, typename Generator>
void
init_result_keepalive(internal::__result_keepalive<T>& ka, sycl::queue& q,
                      std::size_t n, sycl::usm::alloc kind, Generator gen)
{
    constexpr std::size_t offset = 42 * sizeof(int); // offset is divisible by sizeof(int)
    constexpr int poison = 0xDEADBEEF;

    ka.__result_sz = n;
    ka.__kind      = kind;

    if (kind == sycl::usm::alloc::host)
    {
        ka.__offset  = 0;
        T* ptr = sycl::malloc_host<T>(n, q);
        for (std::size_t i = 0; i < n; ++i)
            ptr[i] = gen(i);
        ka.__usm_ptr = ptr;
    }
    else
    {
        ka.__offset = offset;
        auto host_buf = std::shared_ptr<T[]>(std::make_unique<T[]>(offset + n)); // make_shared<T[]> requires C++20
        // poison data in [offset, offset + n)
        int* iptr = reinterpret_cast<int*>(host_buf.get());
        std::fill(iptr, iptr + offset * sizeof(T) / sizeof(int), poison);
        for (std::size_t i = 0; i < n; ++i)
            host_buf[offset + i] = gen(i);

        if (kind == sycl::usm::alloc::device)
        {
            T* ptr = sycl::malloc_device<T>(offset + n, q);
            q.memcpy(ptr, host_buf.get(), (offset + n) * sizeof(T)).wait();
            ka.__usm_ptr = ptr;
        }
        else // sycl::usm::alloc::unknown for sycl::buffer
            ka.__sycl_buf = sycl::buffer<T, 1>{host_buf, sycl::range{offset + n}};
    }
}

} // namespace Test

// Test struct
struct StorageHolderTest
{
    sycl::queue q;
    sycl::usm::alloc scratch_kind;
    sycl::usm::alloc result_kind;
    
    StorageHolderTest(sycl::queue queue) : q(queue)
    {
        // determine which USM type will be used for storage
        hetero::__device_storage<int> ds(q, 100);
        hetero::__result_storage<int> rs(q, 100);
        scratch_kind = ds.__usm_buf ? sycl::usm::alloc::device : sycl::usm::alloc::unknown;
        result_kind = rs.__kind;
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

        Test::store_and_check(ds0, holder);
        Test::store_and_check(ds1, holder);
        Test::store_and_check(ds2, holder);

        EXPECT_EQ(NScratch, holder.scratch_count(), "scratch deposits: final scratch count is incorrect");
        if (scratch_kind != sycl::usm::alloc::unknown)
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

        Test::store_and_check<0>(rs0, holder);
        Test::store_and_check<1>(rs1, holder);
        Test::store_and_check<2>(rs2, holder);
        
        if (result_kind != sycl::usm::alloc::unknown)
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

        Test::store_and_check<0>(cs0, holder);
        Test::store_and_check   (ds,  holder);
        Test::store_and_check<1>(cs1, holder);

        const std::size_t expected_scratch = /*ds*/1 + (result_kind == sycl::usm::alloc::host ? /*cs0&1*/2 : 0);
        EXPECT_EQ(expected_scratch, holder.scratch_count(), "combined deposits: final scratch count is incorrect");

        if (scratch_kind != sycl::usm::alloc::unknown)
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

    void test_copy_result()
    {
        using TupleT = std::tuple<int, long>;
        using HolderT = Test::inspectable_holder<0, TupleT, float, int>;
        auto gen_tuple = [](std::size_t i){ return TupleT{int(i * 37) % 5, long(i * 19 - 32)}; };
        auto gen_float = [](std::size_t i){ return float(i) * 2.17f - 3.1415f; };
        auto gen_int = [](std::size_t i){ return int(i * 3 + 313); };

        auto verify_copy_result = [&](std::size_t n, HolderT& holder)
        {
            std::vector<TupleT> rt(n);
            holder.template __copy_result<0>(rt.data(), n);
            for (std::size_t i = 0; i < n; ++i)
                EXPECT_EQ(gen_tuple(i), rt[i], "copy_result: incorrect tuple data");

            std::vector<float> rf(n);
            holder.template __copy_result<1>(rf.data(), n);
            for (std::size_t i = 0; i < n; ++i)
                EXPECT_EQ(gen_float(i), rf[i], "copy_result: incorrect float data");

            std::vector<int> ri(n);
            holder.template __copy_result<2>(ri.data(), n);
            for (std::size_t i = 0; i < n; ++i)
                EXPECT_EQ(gen_int(i), ri[i], "copy_result: incorrect int data");
        };

        for (std::size_t n : {1, 2, 3, 6, 7})
        {
            HolderT holder{q};
            Test::init_result_keepalive(holder.template result_slot_ref<0>(), q, n, result_kind, gen_tuple);
            Test::init_result_keepalive(holder.template result_slot_ref<1>(), q, n, scratch_kind, gen_float);
            Test::init_result_keepalive(holder.template result_slot_ref<2>(), q, n, sycl::usm::alloc::unknown, gen_int);
            
            if (n == 3) 
            {
                HolderT other(std::move(holder));
                verify_copy_result(n, other);
            }
            else if (n == 6)
            {
                HolderT other{sycl::queue{}};
                other = std::move(holder);
                verify_copy_result(n, other);
            }
            else 
                verify_copy_result(n, holder);
        }

        // Edge case: copy zero elements
        {
            Test::inspectable_holder<0, int> holder{q};
            Test::init_result_keepalive(holder.template result_slot_ref<0>(), q, 4, result_kind,
                                        [](std::size_t i){ return int(i); });
            int sentinel = 42;
            holder.__copy_result<0>(&sentinel, 0);
            EXPECT_EQ(42, sentinel, "copy_result zero: sentinel must be unmodified");
        }
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
        test_copy_result();
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
