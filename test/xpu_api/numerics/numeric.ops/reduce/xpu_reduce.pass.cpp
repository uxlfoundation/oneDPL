//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>
// UNSUPPORTED: c++03, c++11, c++14

// Became constexpr in C++20
// template<class InputIterator>
//     typename iterator_traits<InputIterator>::value_type
//     reduce(InputIterator first, InputIterator last);

#include <oneapi/dpl/numeric>
#include <cassert>

#include "support/utils.h"
#include "support/test_iterators.h"

template <class T>
class KernelTest;

template <typename _T1, typename _T2>
void
ASSERT_EQUAL(_T1&& X, _T2&& Y)
{
    if (X != Y)
        std::cout << "CHECK CORRECTNESS (STL WITH SYCL): fail (" << X << "," << Y << ")" << std::endl;
}

template <class Iter>
void
test(sycl::queue& deviceQueue)
{
    using VT = typename std::iterator_traits<Iter>::value_type;
    int input[] = {1, 2, 3, 4, 5, 6};
    int output[21];
    int ref[21] = {0, 1, 3, 21, 0, 1, 1, 3, 3, 6, 21, 25, 0, 1, 1, 2, 3, 6, 21, 2880, 27};
    sycl::range<1> numOfItems{6};
    {
        sycl::buffer<int, 1> buffer1(input, sycl::range<1>{6});
        sycl::buffer<int, 1> buffer2(output, sycl::range<1>{21});
        deviceQueue.submit(
            [&](sycl::handler& cgh)
            {
                auto in = buffer1.get_access<sycl::access::mode::read>(cgh);
                auto out = buffer2.get_access<sycl::access::mode::write>(cgh);
                cgh.single_task<KernelTest<Iter>>(
                    [=]()
                    {
                        out[0] = dpl::reduce(Iter(&in[0]), Iter(&in[0]));
                        out[1] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 1));
                        out[2] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 2));
                        out[3] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 6));

                        out[4] = dpl::reduce(Iter(&in[0]), Iter(&in[0]), 0);
                        out[5] = dpl::reduce(Iter(&in[0]), Iter(&in[0]), 1);
                        out[6] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 1), 0);
                        out[7] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 1), 2);
                        out[8] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 2), 0);
                        out[9] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 2), 3);
                        out[10] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 6), 0);
                        out[11] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 6), 4);

                        out[12] = dpl::reduce(Iter(&in[0]), Iter(&in[0]), 0, std::plus<>());
                        out[13] = dpl::reduce(Iter(&in[0]), Iter(&in[0]), 1, std::multiplies<>());
                        out[14] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 1), 0, std::plus<>());
                        out[15] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 1), 2, std::multiplies<>());
                        out[16] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 2), 0, std::plus<>());
                        out[17] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 2), 3, std::multiplies<>());
                        out[18] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 6), 0, std::plus<>());
                        out[19] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 6), 4, std::multiplies<>());
                        out[20] = dpl::reduce(Iter(&in[0]), Iter(&in[0] + 6), 0, [](VT x, VT y) { return x + y + 1; });
                    });
            });
    }

    for (size_t idx = 0; idx < 21; ++idx)
    {
        ASSERT_EQUAL(ref[idx], output[idx]);
    }
}

void
test(sycl::queue& deviceQueue)
{
    test<input_iterator<const int*>>(deviceQueue);
    test<forward_iterator<const int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>>(deviceQueue);
    test<random_access_iterator<const int*>>(deviceQueue);
    test<const int*>(deviceQueue);
}

int
main(int, char**)
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    test(deviceQueue);
    return TestUtils::done();
}
