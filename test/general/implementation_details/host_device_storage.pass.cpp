// -*- C++ -*-
//===------------------------------------------------------===//
//
// Copyright (C) 2025 UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#include "support/test_config.h"

#if TEST_DPCPP_BACKEND_PRESENT
// No easy way to only include the relevant internal header
#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#endif

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#include "support/utils_sycl.h"
#include <vector>

template <typename T>
struct CombineResultAndScratch
{
    oneapi::dpl::__par_backend_hetero::__result_storage<T> result;
    oneapi::dpl::__par_backend_hetero::__device_storage<T> scratch;

    CombineResultAndScratch(const sycl::queue& q, std::size_t n_scratch, std::size_t n_result)
        : result(q, n_result), scratch(q, n_scratch) {}

    void
    __copy_result(T* dst, std::size_t sz)
    {
        return result.__copy_result(dst, sz);
    }

    template <sycl::access_mode _AccessMode = sycl::access_mode::read_write>
    friend auto
    __get_accessor(CombineResultAndScratch& rs, sycl::handler& cgh, const sycl::property_list& prop_list = {})
    {
        return __get_accessor<_AccessMode>(rs.scratch, cgh, prop_list);
    }

    template <sycl::access_mode _AccessMode = sycl::access_mode::read_write>
    friend auto
    __get_result_accessor(CombineResultAndScratch& rs, sycl::handler& cgh, const sycl::property_list& prop_list = {})
    {
        return __get_result_accessor<_AccessMode>(rs.result, cgh, prop_list);
    }
};

template<std::size_t, typename...>
struct KernelName; // kernel name

struct Test
{
    sycl::queue q;

    template <template <typename> typename Storage>
    void run_single_kernel(int n_scratch, int n_result)
    {
        using ValueType = int;
        using SingleKernel = KernelName<1, Storage<ValueType>>;

        Storage<ValueType> result_and_scratch(q, n_scratch, n_result);

        q.submit([&](sycl::handler& cgh) {
            auto scratch_acc =
                __get_accessor<sycl::access_mode::read_write>(result_and_scratch, cgh, sycl::property::no_init{});
            auto result_acc =
                __get_result_accessor<sycl::access_mode::write>(result_and_scratch, cgh, sycl::property::no_init{});
            cgh.parallel_for<SingleKernel>(sycl::range<1>(n_scratch), [=](sycl::item<1> wi){
                std::size_t idx = wi.get_linear_id();
                ValueType* scratch = scratch_acc.__data();
                ValueType* result = result_acc.__data();

                scratch[idx] = ValueType(n_scratch - idx); // n_scratch .. 1
                if (idx >= n_scratch - n_result) // the last n_result items
                    result[n_scratch - idx - 1] = scratch[idx] + n_scratch;
            });
        }).wait();

        ValueType single_val;
        result_and_scratch.__copy_result(&single_val, 1);
        EXPECT_EQ(1 + n_scratch, single_val, "Incorrect first value copied");
        
        std::vector<ValueType> expected{n_result};
        ValueType i = 0;
        for (ValueType& v: expected)
            v = ++i + n_scratch;

        std::vector<ValueType> result_host{n_result};
        result_and_scratch.__copy_result(result_host.data(), result_host.size());
        EXPECT_EQ_RANGES(expected, result_host, "Incorrect data copied");
    }

    template <template <typename> typename Storage>
    void run()
    {
        for (int n_scratch = 1, inc = 3; n_scratch < 2000; n_scratch += inc, inc += 2)
            for (int n_result = 1; n_result <= n_scratch && n_result < 10; ++n_result)
            {
                run_single_kernel<Storage>(n_scratch, n_result);
            }
    }

};

#endif // TEST_DPCPP_BACKEND_PRESENT

int main() {
#if TEST_DPCPP_BACKEND_PRESENT
    Test test{TestUtils::get_test_queue()};
    test.run<oneapi::dpl::__par_backend_hetero::__combined_storage>();
    test.run<CombineResultAndScratch>();
#endif
    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
