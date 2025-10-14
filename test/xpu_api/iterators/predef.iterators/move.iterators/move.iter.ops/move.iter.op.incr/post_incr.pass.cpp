//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// move_iterator operator++(int);
//
//  constexpr in C++17

#include "support/test_config.h"

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/type_traits>

#include "support/test_iterators.h"
#include "support/utils.h"
#include "support/test_macros.h"

template <typename It>
constexpr bool is_forward_iterator =
    std::is_base_of_v<std::forward_iterator_tag, typename std::iterator_traits<It>::iterator_category>;

template <class It>
bool
test(It i, It x)
{
    dpl::move_iterator<It> r(i);
#if TEST_STD_VER >= 20
    // dpl::move_iterator post increment operation does not return value if It
    // is not forward iterator
    dpl::move_iterator<It> rr;
    if constexpr (is_forward_iterator<It>)
    {
        rr = r++;
    }
    else
    {
        r++;
    }
#else
    dpl::move_iterator<It> rr = r++;
#endif
    auto ret = (r.base() == x);
#if TEST_STD_VER >= 20
    if constexpr (is_forward_iterator<It>)
#endif
        ret &= (rr.base() == i);
    return ret;
}

bool
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = true;
    {
        sycl::range<1> numOfItems{1};
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                char s[] = "123";
                ret_access[0] &= test(input_iterator<char*>(s), input_iterator<char*>(s + 1));
                ret_access[0] &= test(forward_iterator<char*>(s), forward_iterator<char*>(s + 1));
                ret_access[0] &= test(bidirectional_iterator<char*>(s), bidirectional_iterator<char*>(s + 1));
                ret_access[0] &= test(random_access_iterator<char*>(s), random_access_iterator<char*>(s + 1));
                ret_access[0] &= test(s, s + 1);

                {
                    constexpr const char* p = "123456789";
                    typedef dpl::move_iterator<const char*> MI;
                    constexpr MI it1 = dpl::make_move_iterator(p);
                    constexpr MI it2 = dpl::make_move_iterator(p + 1);
                    static_assert(it1 != it2);
                    constexpr MI it3 = dpl::make_move_iterator(p)++;
                    static_assert(it1 == it3);
                    static_assert(it2 != it3);
                }
            });
        });
    }
    return ret;
}

int
main()
{
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of move_iterator and operator++() in kernel_test()");

    return TestUtils::done();
}
