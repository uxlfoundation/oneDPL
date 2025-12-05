// -*- C++ -*-
//===-- uninitialized_fill_destroy.pass.cpp -------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"

#if  !defined(_PSTL_TEST_UNITIALIZED_FILL) && !defined(_PSTL_TEST_UNITIALIZED_FILL_N) &&\
     !defined(_PSTL_TEST_UNITIALIZED_DESTROY) && !defined(_PSTL_TEST_UNITIALIZED_DESTROY_N)
#define _PSTL_TEST_UNITIALIZED_FILL
#define _PSTL_TEST_UNITIALIZED_FILL_N
#define _PSTL_TEST_UNITIALIZED_DESTROY
#define _PSTL_TEST_UNITIALIZED_DESTROY_N
#endif

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(memory)

#include "support/utils.h"

#include <memory>
#include <cstdlib>

using namespace TestUtils;

template <typename Type>
struct test_uninitialized_fill
{
    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, const T& in, ::std::size_t n, ::std::false_type)
    {
        using namespace std;

        uninitialized_fill(std::forward<Policy>(exec), first, last, in);
        size_t count = count_if(first, last, [&in](T& x) -> bool { return x == in; });
        EXPECT_TRUE(n == count, "wrong work of uninitialized_fill");

        destroy(oneapi::dpl::execution::seq, first, last);
    }

    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, const T& in, ::std::size_t n, ::std::true_type)
    {
        using namespace std;

        uninitialized_fill(std::forward<Policy>(exec), first, last, in);
        size_t count = count_if(first, last, [&in](T& x) -> bool { return x == in; });
        EXPECT_EQ(n, count, "wrong work of uninitialized_fill");
    }
};

template <typename Type>
struct test_uninitialized_fill_n
{
    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, const T& in, ::std::size_t n, ::std::false_type)
    {
        using namespace std;

        auto res = uninitialized_fill_n(std::forward<Policy>(exec), first, n, in);
        EXPECT_TRUE(res == last, "wrong result of uninitialized_fill_n");
        size_t count = count_if(first, last, [&in](T& x) -> bool { return x == in; });
        EXPECT_TRUE(n == count, "wrong work of uninitialized_fill_n");

        destroy_n(oneapi::dpl::execution::seq, first, n);
    }
    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, const T& in, ::std::size_t n, ::std::true_type)
    {
        using namespace std;

        auto res = uninitialized_fill_n(std::forward<Policy>(exec), first, n, in);
        size_t count = count_if(first, last, [&in](T& x) -> bool { return x == in; });
        EXPECT_EQ(n, count, "wrong work of uninitialized_fill_n");
        EXPECT_TRUE(res == last, "wrong result of uninitialized_fill_n");
    }
};

template <typename Type>
struct test_destroy
{
    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, const T& in, ::std::size_t /* n */, ::std::false_type)
    {
        using namespace std;

        T::SetCount(0);
#if _PSTL_STD_UNINITIALIZED_FILL_BROKEN
        uninitialized_fill(oneapi::dpl::execution::seq, first, last, in);
#else
        uninitialized_fill(first, last, in);
#endif
        destroy(std::forward<Policy>(exec), first, last);
        EXPECT_TRUE(T::Count() == 0, "wrong work of destroy");
    }

    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, const T& in, ::std::size_t /* n */, ::std::true_type)
    {
        using namespace std;

#if _PSTL_STD_UNINITIALIZED_FILL_BROKEN
        uninitialized_fill(oneapi::dpl::execution::seq, first, last, in);
#else
        uninitialized_fill(first, last, in);
#endif
        destroy(std::forward<Policy>(exec), first, last);
        size_t count = count_if(first, last, [&in](T& x) -> bool { return x != in; });
        size_t tmp_n = 0;
        EXPECT_EQ(tmp_n, count, "wrong work of destroy");
    }
};

template <typename Type>
struct test_destroy_n
{
    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, const T& in, ::std::size_t n, ::std::false_type)
    {
        using namespace std;

        T::SetCount(0);
#if _PSTL_STD_UNINITIALIZED_FILL_BROKEN
        uninitialized_fill_n(oneapi::dpl::execution::seq, first, n, in);
#else
        uninitialized_fill(first, last, in);
#endif
        auto dres = destroy_n(std::forward<Policy>(exec), first, n);
        EXPECT_TRUE(dres == last, "wrong result of destroy_n");
        EXPECT_TRUE(T::Count() == 0, "wrong work of destroy_n");
    }

    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, const T& in, ::std::size_t n, ::std::true_type)
    {
        using namespace std;

#if _PSTL_STD_UNINITIALIZED_FILL_BROKEN
        uninitialized_fill_n(oneapi::dpl::execution::seq, first, n, in);
#else
        uninitialized_fill(first, last, in);
#endif
        auto dres = destroy_n(std::forward<Policy>(exec), first, n);
        EXPECT_TRUE(dres == last, "wrong result of destroy_n");
        size_t count = count_if(first, last, [&in](T& x) -> bool { return x != in; });
        size_t tmp_n = 0;
        EXPECT_EQ(tmp_n, count, "wrong work of destroy");
    }
};

template <typename T>
void
test_uninitialized_fill_destroy_by_type()
{
    ::std::size_t N = 100000;
    for (size_t n = 0; n <= N; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
#if !TEST_DPCPP_BACKEND_PRESENT
        // We use malloc / free instead of a typical approach with operator new T[] and a smart pointer (which calls
        // operator delete[] within its destructor) to avoid double creation/destroy the elements of type T in the test.
        auto free_allocation = [](auto ptr) { ::std::free(ptr); };
        ::std::unique_ptr<T[], decltype(free_allocation)> p(static_cast<T*>(::std::malloc(sizeof(T) * n)),
                                                            free_allocation);
        auto p_begin = p.get();
#else
        Sequence<T> p(n, [](size_t){ return T{}; });
        auto p_begin = p.begin();
#endif
        auto p_end = ::std::next(p_begin, n);
#ifdef _PSTL_TEST_UNITIALIZED_FILL
        invoke_on_all_policies<>()(test_uninitialized_fill<T>(), p_begin, p_end, T(), n,
                                   ::std::is_trivial<T>());
#endif
#ifdef _PSTL_TEST_UNITIALIZED_FILL_N
        invoke_on_all_policies<>()(test_uninitialized_fill_n<T>(), p_begin, p_end, T(), n,
                                   ::std::is_trivial<T>());
#endif
#if !TEST_DPCPP_BACKEND_PRESENT
        // SYCL kernel cannot call through a function pointer
#ifdef _PSTL_TEST_UNITIALIZED_DESTROY
        invoke_on_all_policies<>()(test_destroy<T>(), p_begin, p_end, T(), n,
                                   ::std::is_trivial<T>());
#endif
#ifdef _PSTL_TEST_UNITIALIZED_DESTROY_N
        invoke_on_all_policies<>()(test_destroy_n<T>(), p_begin, p_end, T(), n,
                                   ::std::is_trivial<T>());
#endif
#endif
    }
}

void test_empty_list_initialization_for_uninitialized_fill()
{
    constexpr std::size_t size = 10;
    const auto deleter = [](auto ptr) { operator delete(ptr); };
    using deleter_type = decltype(deleter);
    {
        using value_type = TestUtils::Wrapper<int>;
        value_type::SetCount(0);
        std::unique_ptr<value_type, deleter_type> ptr((value_type*)operator new(sizeof(value_type) * size), deleter);
        oneapi::dpl::uninitialized_fill(oneapi::dpl::execution::seq, ptr.get(), ptr.get() + size, {1});
        EXPECT_TRUE(std::count_if(ptr.get(), ptr.get() + size, [](auto x) { return (*x.get_my_field()) == 1; }) == size,
                    "a sequence is not filled properly by oneapi::dpl::uninitialized_fill with `seq` policy");
        EXPECT_TRUE(value_type::Count() == 10, "wrong effect of calling `oneapi::dpl::uninitialized_fill with `seq` policy");
        oneapi::dpl::destroy(oneapi::dpl::execution::seq, ptr.get(), ptr.get() + size);
        EXPECT_TRUE(value_type::Count() == 0, "wrong effect of calling `oneapi::dpl::destroy with `seq` policy");
    }
    {
        using value_type = TestUtils::Wrapper<int>;
        value_type::SetCount(0);
        std::unique_ptr<TestUtils::Wrapper<int>, deleter_type> ptr((value_type*)operator new(sizeof(value_type) * size), deleter);
        oneapi::dpl::uninitialized_fill(oneapi::dpl::execution::unseq, ptr.get(), ptr.get() + size, {1});
        EXPECT_TRUE(std::count_if(ptr.get(), ptr.get() + size, [](auto x) { return (*x.get_my_field()) == 1; }) == size,
                    "a sequence is not filled properly by oneapi::dpl::uninitialized_fill with `unseq` policy");
        EXPECT_TRUE(value_type::Count() == 10, "wrong effect of calling `oneapi::dpl::uninitialized_fill with `unseq` policy");
        oneapi::dpl::destroy(oneapi::dpl::execution::unseq, ptr.get(), ptr.get() + size);
        EXPECT_TRUE(value_type::Count() == 0, "wrong effect of calling `oneapi::dpl::destroy with `unseq` policy");
    }

    {
        using value_type = TestUtils::Wrapper<TestUtils::DefaultInitializedToOne>;
        value_type::SetCount(0);
        std::unique_ptr<value_type, deleter_type> ptr_custom{(value_type*)operator new(sizeof(value_type) * size), deleter};
        oneapi::dpl::uninitialized_fill(oneapi::dpl::execution::par, ptr_custom.get(), ptr_custom.get() + size, {});
        EXPECT_TRUE(std::count_if(ptr_custom.get(), ptr_custom.get() + size, [](auto x) { return (*x.get_my_field()) == TestUtils::DefaultInitializedToOne{1}; }) == size,
                    "a sequence is not filled properly by oneapi::dpl::uninitialized_fill with `par` policy");
        EXPECT_TRUE(value_type::Count() == 10, "wrong effect of calling `oneapi::dpl::uninitialized_fill with `par` policy");
        oneapi::dpl::destroy(oneapi::dpl::execution::par, ptr_custom.get(), ptr_custom.get() + size);
        EXPECT_TRUE(value_type::Count() == 0, "wrong effect of calling `oneapi::dpl::destroy with `par` policy");
    }
    {
        using value_type = TestUtils::Wrapper<TestUtils::DefaultInitializedToOne>;
        value_type::SetCount(0);
        std::unique_ptr<value_type, deleter_type> ptr_custom{(value_type*)operator new(sizeof(value_type) * size), deleter};
        oneapi::dpl::uninitialized_fill(oneapi::dpl::execution::par_unseq, ptr_custom.get(), ptr_custom.get() + size, {});
        EXPECT_TRUE(std::count_if(ptr_custom.get(), ptr_custom.get() + size, [](auto x) { return (*x.get_my_field()) == TestUtils::DefaultInitializedToOne{1}; }) == size,
                    "a sequence is not filled properly by oneapi::dpl::uninitialized_fill with `par_unseq` policy");
        EXPECT_TRUE(value_type::Count() == 10, "wrong effect of calling `oneapi::dpl::uninitialized_fill with `par_unseq` policy");
        oneapi::dpl::destroy(oneapi::dpl::execution::par_unseq, ptr_custom.get(), ptr_custom.get() + size);
        EXPECT_TRUE(value_type::Count() == 0, "wrong effect of calling `oneapi::dpl::destroy with `par_unseq` policy");
    }
#if TEST_DPCPP_BACKEND_PRESENT
    auto usm_deleter = [] (auto ptr) { sycl::free(ptr, oneapi::dpl::execution::dpcpp_default.queue()); };
    using usm_deleter_type = decltype(usm_deleter);
    std::unique_ptr<int, usm_deleter_type> ptr{sycl::malloc_shared<int>(size, oneapi::dpl::execution::dpcpp_default.queue()), usm_deleter};
    oneapi::dpl::uninitialized_fill(oneapi::dpl::execution::dpcpp_default, ptr.get(), ptr.get() + size, {1});
    EXPECT_TRUE(std::count(ptr.get(), ptr.get() + size, 1) == size, "a sequence is not filled properly by oneapi::dpl::uninitialized_fill with `device_policy` policy");
    // no need to call destroy for a trivial type
#endif
}

void test_empty_list_initialization_for_uninitialized_fill_n()
{
    constexpr std::size_t size = 10;
    const auto deleter = [](auto ptr) { operator delete(ptr); };
    using deleter_type = decltype(deleter);
    {
        using value_type = TestUtils::Wrapper<int>;
        value_type::SetCount(0);
        std::unique_ptr<value_type, deleter_type> ptr((value_type*)operator new(sizeof(value_type) * size), deleter);
        oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::seq, ptr.get(), size, {1});
        EXPECT_TRUE(std::count_if(ptr.get(), ptr.get() + size, [](auto x) { return (*x.get_my_field()) == 1; }) == size,
                    "a sequence is not filled properly by oneapi::dpl::uninitialized_fill_n with `seq` policy");
        EXPECT_TRUE(value_type::Count() == 10, "wrong effect of calling `oneapi::dpl::uninitialized_fill_n with `seq` policy");
        oneapi::dpl::destroy(oneapi::dpl::execution::seq, ptr.get(), ptr.get() + size);
        EXPECT_TRUE(value_type::Count() == 0, "wrong effect of calling `oneapi::dpl::destroy with `seq` policy");
    }
    {
        using value_type = TestUtils::Wrapper<int>;
        value_type::SetCount(0);
        std::unique_ptr<TestUtils::Wrapper<int>, deleter_type> ptr((value_type*)operator new(sizeof(value_type) * size), deleter);
        oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::unseq, ptr.get(), size, {1});
        EXPECT_TRUE(std::count_if(ptr.get(), ptr.get() + size, [](auto x) { return (*x.get_my_field()) == 1; }) == size,
                    "a sequence is not filled properly by oneapi::dpl::uninitialized_fill_n with `unseq` policy");
        EXPECT_TRUE(value_type::Count() == 10, "wrong effect of calling `oneapi::dpl::uninitialized_fill_n with `unseq` policy");
        oneapi::dpl::destroy(oneapi::dpl::execution::unseq, ptr.get(), ptr.get() + size);
        EXPECT_TRUE(value_type::Count() == 0, "wrong effect of calling `oneapi::dpl::destroy with `unseq` policy");
    }

    {
        using value_type = TestUtils::Wrapper<TestUtils::DefaultInitializedToOne>;
        value_type::SetCount(0);
        std::unique_ptr<value_type, deleter_type> ptr_custom{(value_type*)operator new(sizeof(value_type) * size), deleter};
        oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::par, ptr_custom.get(), size, {});
        EXPECT_TRUE(std::count_if(ptr_custom.get(), ptr_custom.get() + size, [](auto x) { return (*x.get_my_field()) == TestUtils::DefaultInitializedToOne{1}; }) == size,
                    "a sequence is not filled properly by oneapi::dpl::uninitialized_fill_n with `par` policy");
        EXPECT_TRUE(value_type::Count() == 10, "wrong effect of calling `oneapi::dpl::uninitialized_fill_n with `par` policy");
        oneapi::dpl::destroy(oneapi::dpl::execution::par, ptr_custom.get(), ptr_custom.get() + size);
        EXPECT_TRUE(value_type::Count() == 0, "wrong effect of calling `oneapi::dpl::destroy with `par` policy");
    }
    {
        using value_type = TestUtils::Wrapper<TestUtils::DefaultInitializedToOne>;
        value_type::SetCount(0);
        std::unique_ptr<value_type, deleter_type> ptr_custom{(value_type*)operator new(sizeof(value_type) * size), deleter};
        oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::par_unseq, ptr_custom.get(), size, {});
        EXPECT_TRUE(std::count_if(ptr_custom.get(), ptr_custom.get() + size, [](auto x) { return (*x.get_my_field()) == TestUtils::DefaultInitializedToOne{1}; }) == size,
                    "a sequence is not filled properly by oneapi::dpl::uninitialized_fill_n with `par_unseq` policy");
        EXPECT_TRUE(value_type::Count() == 10, "wrong effect of calling `oneapi::dpl::uninitialized_fill_n with `par_unseq` policy");
        oneapi::dpl::destroy(oneapi::dpl::execution::par_unseq, ptr_custom.get(), ptr_custom.get() + size);
        EXPECT_TRUE(value_type::Count() == 0, "wrong effect of calling `oneapi::dpl::destroy with `par_unseq` policy");
    }
#if TEST_DPCPP_BACKEND_PRESENT
    auto usm_deleter = [] (auto ptr) { sycl::free(ptr, oneapi::dpl::execution::dpcpp_default.queue()); };
    using usm_deleter_type = decltype(usm_deleter);
    std::unique_ptr<int, usm_deleter_type> ptr{sycl::malloc_shared<int>(size, oneapi::dpl::execution::dpcpp_default.queue()), usm_deleter};
    oneapi::dpl::uninitialized_fill_n(oneapi::dpl::execution::dpcpp_default, ptr.get(), size, {1});
    EXPECT_TRUE(std::count(ptr.get(), ptr.get() + size, 1) == size, "a sequence is not filled properly by oneapi::dpl::uninitialized_fill with `device_policy` policy");
    // no need to call destroy for a trivial type
#endif
}

int
main()
{
    // for trivial types
    test_uninitialized_fill_destroy_by_type<std::int32_t>();
    test_uninitialized_fill_destroy_by_type<float64_t>();

#if !TEST_DPCPP_BACKEND_PRESENT
    // for user-defined types
    test_uninitialized_fill_destroy_by_type<Wrapper<::std::string>>();
    test_uninitialized_fill_destroy_by_type<Wrapper<std::int8_t*>>();
#endif

    test_empty_list_initialization_for_uninitialized_fill();
    test_empty_list_initialization_for_uninitialized_fill_n();

    return done();
}
