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
#include <oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h>
#include "support/utils_sycl_defs.h"

#include <vector>

struct Test
{
    sycl::queue q;

    template <template <typename> typename Storage>
    void run_single_kernel(int n_scratch, int n_result)
    {
        using ValueType = int;
        struct SingleKernel; // kernel name

        Storage<ValueType> result_and_scratch{q, n_scratch, n_result};

        q.submit([&](sycl::handler& cgh) {
            auto scratch_acc =
                __get_accessor<sycl::access_mode::read_write>(result_and_scratch, cgh, sycl::property::no_init{});
            auto result_acc =
                __get_result_accessor<sycl::access_mode::write>(result_and_scratch, cgh, sycl::property::no_init{});
            cgh.parallel_for<SingleKernel>(sycl::range<1>{n_scratch}, [=](sycl::item<1> wi){
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

#include "support/utils.h"

int main() {
#if TEST_DPCPP_BACKEND_PRESENT
    Test test{};
    test.run<oneapi::dpl::__par_backend_hetero::__combined_storage>();
#endif
    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
