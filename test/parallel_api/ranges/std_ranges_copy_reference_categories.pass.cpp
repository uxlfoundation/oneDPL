// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// indirectly_copyable<I, O> grants only *out = *in with *in of type iter_reference_t<I>. The device copy_if,
// remove_copy_if, partition_copy and unique_copy must therefore assign every written element with the value category
// and the constness of the reference type of the input. Each input below is paired with an output that accepts
// exactly that reference type and rejects the others:
//   - a mutable input (In&)          and an output assignable only from In&,
//   - a const input (const In&)      and an output that rejects In&,
//   - a prvalue input (In)           and an output assignable only from In&&,
//   - a const prvalue input (const In) and an output that rejects In&&,
//   - a volatile input (volatile int&) and an output that rejects int&,
//   - a proxy input (a by-value proxy convertible to const int&, the value type being int) and an output that
//     rejects int&&,
//   - a proxy input (a by-value proxy convertible to int, the value type being int) and an output assignable only
//     from int&&.
// Every algorithm runs on a small input (the single-group path) and a large one (the reduce-then-scan path), with an
// output that is large enough and with one that is too small, and the results and the returned positions are checked.
// Besides, reverse_copy, whose vectorized path stages the input in a local array, copies const elements into a proxy
// output that rejects a non-const lvalue.

#include "support/test_config.h"

#if _ENABLE_STD_RANGES_TESTING && TEST_DPCPP_BACKEND_PRESENT
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/ranges>

#include <ranges>
#include <span>
#include <vector>
#endif

#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING && TEST_DPCPP_BACKEND_PRESENT

struct In
{
    int v;
};

struct MutLvalueOut
{
    int v;
    MutLvalueOut&
    operator=(In& x)
    {
        v = x.v;
        return *this;
    }
    MutLvalueOut&
    operator=(const In&) = delete;
    MutLvalueOut&
    operator=(In&&) = delete;
};

struct ConstLvalueOut
{
    int v;
    ConstLvalueOut&
    operator=(const In& x)
    {
        v = x.v;
        return *this;
    }
    ConstLvalueOut&
    operator=(In&) = delete;
};

struct RvalueOut
{
    int v;
    RvalueOut&
    operator=(In&& x)
    {
        v = x.v;
        return *this;
    }
    RvalueOut&
    operator=(const In&) = delete;
    RvalueOut&
    operator=(In&) = delete;
};

struct VolatileOut
{
    int v;
    VolatileOut&
    operator=(const int& x)
    {
        v = x;
        return *this;
    }
    VolatileOut&
    operator=(const volatile int& x)
    {
        v = x;
        return *this;
    }
    VolatileOut&
    operator=(int&) = delete;
};

struct ProxyOut
{
    int v;
    ProxyOut&
    operator=(const int& x)
    {
        v = x;
        return *this;
    }
    ProxyOut&
    operator=(int&&) = delete;
};

struct ProxyRef
{
    int v;
    operator const int&() const { return v; }
};

struct ProxyValueOut
{
    int v;
    ProxyValueOut&
    operator=(int&& x)
    {
        v = x;
        return *this;
    }
    ProxyValueOut&
    operator=(const int&) = delete;
    ProxyValueOut&
    operator=(int&) = delete;
};

struct ProxyValue
{
    int v;
    operator int() const { return v; }
};

// A random access iterator over 0, 1, 2, ... whose value type is int and whose reference type is a by-value proxy
template <typename _Proxy>
struct ProxyIt
{
    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using iterator_concept = std::random_access_iterator_tag;

    int i = 0;

    _Proxy operator*() const { return _Proxy{i}; }
    _Proxy operator[](difference_type d) const { return _Proxy{int(i + d)}; }
    ProxyIt& operator++() { ++i; return *this; }
    ProxyIt operator++(int) { ProxyIt t = *this; ++i; return t; }
    ProxyIt& operator--() { --i; return *this; }
    ProxyIt operator--(int) { ProxyIt t = *this; --i; return t; }
    ProxyIt& operator+=(difference_type d) { i += d; return *this; }
    ProxyIt& operator-=(difference_type d) { i -= d; return *this; }
    friend ProxyIt operator+(ProxyIt it, difference_type d) { return it += d; }
    friend ProxyIt operator+(difference_type d, ProxyIt it) { return it += d; }
    friend ProxyIt operator-(ProxyIt it, difference_type d) { return it -= d; }
    friend difference_type operator-(ProxyIt a, ProxyIt b) { return a.i - b.i; }
    friend bool operator==(ProxyIt a, ProxyIt b) { return a.i == b.i; }
    friend auto operator<=>(ProxyIt a, ProxyIt b) { return a.i <=> b.i; }
};
static_assert(std::random_access_iterator<ProxyIt<ProxyRef>>);
static_assert(std::random_access_iterator<ProxyIt<ProxyValue>>);

struct ShortOutProxy
{
    short* p;
    const ShortOutProxy&
    operator=(const short& x) const
    {
        *p = x;
        return *this;
    }
    const ShortOutProxy&
    operator=(short&) const = delete;
    operator short() const { return *p; }
};

// A random access iterator over short whose reference type is ShortOutProxy
struct ProxyOutIt
{
    using value_type = short;
    using difference_type = std::ptrdiff_t;
    using iterator_concept = std::random_access_iterator_tag;

    short* p = nullptr;

    ShortOutProxy operator*() const { return ShortOutProxy{p}; }
    ShortOutProxy operator[](difference_type d) const { return ShortOutProxy{p + d}; }
    ProxyOutIt& operator++() { ++p; return *this; }
    ProxyOutIt operator++(int) { ProxyOutIt t = *this; ++p; return t; }
    ProxyOutIt& operator--() { --p; return *this; }
    ProxyOutIt operator--(int) { ProxyOutIt t = *this; --p; return t; }
    ProxyOutIt& operator+=(difference_type d) { p += d; return *this; }
    ProxyOutIt& operator-=(difference_type d) { p -= d; return *this; }
    friend ProxyOutIt operator+(ProxyOutIt it, difference_type d) { return it += d; }
    friend ProxyOutIt operator+(difference_type d, ProxyOutIt it) { return it += d; }
    friend ProxyOutIt operator-(ProxyOutIt it, difference_type d) { return it -= d; }
    friend difference_type operator-(ProxyOutIt a, ProxyOutIt b) { return a.p - b.p; }
    friend bool operator==(ProxyOutIt a, ProxyOutIt b) { return a.p == b.p; }
    friend auto operator<=>(ProxyOutIt a, ProxyOutIt b) { return a.p <=> b.p; }
};
static_assert(std::random_access_iterator<ProxyOutIt>);
static_assert(std::indirectly_writable<ProxyOutIt, const short&>);

int
value_of(const In& x)
{
    return x.v;
}

int
value_of(int x)
{
    return x;
}

struct ConstRvalueOut
{
    int v;
    ConstRvalueOut&
    operator=(const In& x)
    {
        v = x.v;
        return *this;
    }
    ConstRvalueOut&
    operator=(In&&) = delete;
};

// The input values are 0, 1, ..., n - 1. An algorithm writes the values `kept` in this order; with an output of size
// `out_size` the first `count` of them are written and the input stops at `stop`.
struct expected_t
{
    std::vector<int> kept;
    int count;
    int stop;
};

template <typename _Keep>
expected_t
expect(int n, int out_size, _Keep keep)
{
    expected_t e{{}, 0, n};
    for (int i = 0; i < n; ++i)
    {
        if (!keep(i))
            continue;
        if (int(e.kept.size()) == out_size)
        {
            e.stop = i;
            break;
        }
        e.kept.push_back(i);
    }
    e.count = e.kept.size();
    return e;
}

template <typename _Out>
bool
same_values(const _Out* out, const std::vector<int>& expected)
{
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (out[i].v != expected[i])
            return false;
    return true;
}

// A distinct kernel name for every algorithm and output type (the input types contain lambdas, so they cannot be used)
template <typename _Out, int _Algo>
class kernel_name;

template <typename _Out, typename _Policy, typename _Input>
void
test_input(_Policy&& policy, _Input in, int n, int out_size, const char* kind)
{
    auto queue = policy.queue();
    _Out* out1 = sycl::malloc_shared<_Out>(n, queue);
    _Out* out2 = sycl::malloc_shared<_Out>(n, queue);
    std::span<_Out> r1(out1, out_size), r2(out2, out_size);
    const std::string msg = std::string(kind) + ", n = " + std::to_string(n) + ", out_size = " + std::to_string(out_size);

    auto is_even = [](const auto& x) { return value_of(x) % 2 == 0; };

    {
        const expected_t e = expect(n, out_size, [](int i) { return i % 2 == 0; });
        auto res = oneapi::dpl::ranges::copy_if(TestUtils::make_new_policy<kernel_name<_Out, 0>>(policy), in, r1, is_even);
        EXPECT_EQ(e.stop, int(res.in - std::ranges::begin(in)), ("copy_if: wrong input position, " + msg).c_str());
        EXPECT_EQ(e.count, int(res.out - r1.begin()), ("copy_if: wrong output position, " + msg).c_str());
        EXPECT_TRUE(same_values(out1, e.kept), ("copy_if: wrong output, " + msg).c_str());
    }
    {
        const expected_t e = expect(n, out_size, [](int i) { return i % 2 != 0; });
        auto res = oneapi::dpl::ranges::remove_copy_if(TestUtils::make_new_policy<kernel_name<_Out, 1>>(policy), in, r1, is_even);
        EXPECT_EQ(e.stop, int(res.in - std::ranges::begin(in)), ("remove_copy_if: wrong input position, " + msg).c_str());
        EXPECT_EQ(e.count, int(res.out - r1.begin()), ("remove_copy_if: wrong output position, " + msg).c_str());
        EXPECT_TRUE(same_values(out1, e.kept), ("remove_copy_if: wrong output, " + msg).c_str());
    }
    {
        // 2 * k and 2 * k + 1 are equivalent, so the even values are kept
        const expected_t e = expect(n, out_size, [](int i) { return i % 2 == 0; });
        auto res = oneapi::dpl::ranges::unique_copy(
            TestUtils::make_new_policy<kernel_name<_Out, 2>>(policy), in, r1,
            [](const auto& a, const auto& b) { return value_of(a) / 2 == value_of(b) / 2; });
        EXPECT_EQ(e.stop, int(res.in - std::ranges::begin(in)), ("unique_copy: wrong input position, " + msg).c_str());
        EXPECT_EQ(e.count, int(res.out - r1.begin()), ("unique_copy: wrong output position, " + msg).c_str());
        EXPECT_TRUE(same_values(out1, e.kept), ("unique_copy: wrong output, " + msg).c_str());
    }
    {
        // the even values go to the first output, the odd ones to the second; the input stops at the first value
        // that does not fit into its output
        std::vector<int> evens, odds;
        int stop = n;
        for (int i = 0; i < n; ++i)
        {
            std::vector<int>& to = (i % 2 == 0) ? evens : odds;
            if (int(to.size()) == out_size)
            {
                stop = i;
                break;
            }
            to.push_back(i);
        }
        auto res = oneapi::dpl::ranges::partition_copy(TestUtils::make_new_policy<kernel_name<_Out, 3>>(policy), in, r1, r2, is_even);
        EXPECT_EQ(stop, int(res.in - std::ranges::begin(in)), ("partition_copy: wrong input position, " + msg).c_str());
        EXPECT_EQ(int(evens.size()), int(res.out1 - r1.begin()),
                  ("partition_copy: wrong first output position, " + msg).c_str());
        EXPECT_EQ(int(odds.size()), int(res.out2 - r2.begin()),
                  ("partition_copy: wrong second output position, " + msg).c_str());
        EXPECT_TRUE(same_values(out1, evens), ("partition_copy: wrong first output, " + msg).c_str());
        EXPECT_TRUE(same_values(out2, odds), ("partition_copy: wrong second output, " + msg).c_str());
    }

    sycl::free(out1, queue);
    sycl::free(out2, queue);
}

template <typename _Policy>
void
test_reference_categories(_Policy&& policy, int n)
{
    auto queue = policy.queue();
    In* data = sycl::malloc_shared<In>(n, queue);
    int* ints = sycl::malloc_shared<int>(n, queue);
    for (int i = 0; i < n; ++i)
    {
        data[i].v = i;
        ints[i] = i;
    }

    auto mut_in = std::span<In>(data, n);
    auto const_in = std::span<const In>(data, n);
    auto prvalue_in = std::views::iota(0, n) | std::views::transform([](int i) -> In { return In{i}; });
    auto const_prvalue_in = std::views::iota(0, n) | std::views::transform([](int i) -> const In { return In{i}; });
    auto volatile_in = std::span<volatile int>(ints, n);
    auto proxy_in = std::ranges::subrange<ProxyIt<ProxyRef>>(ProxyIt<ProxyRef>{0}, ProxyIt<ProxyRef>{n});
    auto proxy_value_in = std::ranges::subrange<ProxyIt<ProxyValue>>(ProxyIt<ProxyValue>{0}, ProxyIt<ProxyValue>{n});
    static_assert(std::is_same_v<std::ranges::range_reference_t<decltype(mut_in)>, In&>);
    static_assert(std::is_same_v<std::ranges::range_reference_t<decltype(const_in)>, const In&>);
    static_assert(std::is_same_v<std::ranges::range_reference_t<decltype(prvalue_in)>, In>);
    static_assert(std::is_same_v<std::ranges::range_reference_t<decltype(const_prvalue_in)>, const In>);
    static_assert(std::is_same_v<std::ranges::range_reference_t<decltype(volatile_in)>, volatile int&>);
    static_assert(std::is_same_v<std::ranges::range_reference_t<decltype(proxy_in)>, ProxyRef>);
    static_assert(std::is_same_v<std::ranges::range_value_t<decltype(proxy_in)>, int>);
    static_assert(std::is_same_v<std::ranges::range_reference_t<decltype(proxy_value_in)>, ProxyValue>);

    // a large enough output, and one that holds only a quarter of the input
    for (int out_size : {n, n / 4})
    {
        test_input<MutLvalueOut>(policy, mut_in, n, out_size, "mutable input");
        test_input<ConstLvalueOut>(policy, const_in, n, out_size, "const input");
        test_input<RvalueOut>(policy, prvalue_in, n, out_size, "prvalue input");
        test_input<ConstRvalueOut>(policy, const_prvalue_in, n, out_size, "const prvalue input");
        test_input<VolatileOut>(policy, volatile_in, n, out_size, "volatile input");
        test_input<ProxyOut>(policy, proxy_in, n, out_size, "proxy input");
        test_input<ProxyValueOut>(policy, proxy_value_in, n, out_size, "proxy-to-value input");
    }

    sycl::free(data, queue);
    sycl::free(ints, queue);
}

template <typename _Policy>
void
test_staged_store(_Policy&& policy, int n)
{
    auto queue = policy.queue();
    short* in = sycl::malloc_shared<short>(n, queue);
    short* out = sycl::malloc_shared<short>(n, queue);
    for (int i = 0; i < n; ++i)
        in[i] = short(i % 1000);

    auto res = oneapi::dpl::ranges::reverse_copy(
        TestUtils::make_new_policy<kernel_name<ShortOutProxy, 0>>(policy), std::span<const short>(in, n),
        std::ranges::subrange<ProxyOutIt>(ProxyOutIt{out}, ProxyOutIt{out + n}));
    EXPECT_EQ(n, int(res.out.p - out), ("reverse_copy: wrong output position, n = " + std::to_string(n)).c_str());
    bool same = true;
    for (int i = 0; i < n; ++i)
        same = same && out[i] == in[n - 1 - i];
    EXPECT_TRUE(same, ("reverse_copy: wrong output, n = " + std::to_string(n)).c_str());

    sycl::free(in, queue);
    sycl::free(out, queue);
}

#endif // _ENABLE_STD_RANGES_TESTING && TEST_DPCPP_BACKEND_PRESENT

int
main()
{
    bool bProcessed = false;

#if _ENABLE_STD_RANGES_TESTING && TEST_DPCPP_BACKEND_PRESENT
    auto policy = TestUtils::get_dpcpp_test_policy();
    if (policy.queue().get_device().has(sycl::aspect::usm_shared_allocations))
    {
        test_reference_categories(policy, 100);    // the single-group path
        test_reference_categories(policy, 100000); // the reduce-then-scan path
        test_staged_store(policy, 100);
        test_staged_store(policy, 100000);

        bProcessed = true;
    }
#endif

    return TestUtils::done(bProcessed);
}
