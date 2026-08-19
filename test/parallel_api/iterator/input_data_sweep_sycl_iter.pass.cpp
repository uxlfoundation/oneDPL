// -*- C++ -*-
//===-- input_data_sweep_sycl_iter.pass.cpp -------------------------------===//
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

#include "support/utils.h"
#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(iterator)

#include "input_data_sweep.h"

#include "support/utils_invoke.h"

//This test is written without indirection from invoke_on_all_hetero_policies to make clear exactly which types
// are being tested, and to limit the number of types to be within reason.

#if TEST_DPCPP_BACKEND_PRESENT

// The value written into the visited elements by the checks below.
template <typename T>
inline constexpr T new_value = static_cast<T>(-333);

// A transform_iterator built with this functor swallows any write made through it, its base sequence is never written.
inline constexpr auto drop_write = [](auto&&) { return std::ignore; };

// A permutation_iterator built with this map functor sends every access to the 0th element of its base sequence,
// leaving every other element of it unwritten.
inline constexpr auto map_to_zero = [](auto) { return 0; };

// Source sequence of the copy below: new_value<T> repeated, so that the elements which are written end up holding the
// same value no matter which source element a write comes from.
template <typename T>
auto
make_constant_source()
{
    return oneapi::dpl::make_transform_iterator(oneapi::dpl::counting_iterator<int>(0),
                                                [](auto) { return new_value<T>; });
}

// Copies a constant sequence into the wrapped sequence. copy requests no_init for its output internally.
template <typename T, bool __zipped>
struct run_copy
{
    static constexpr const char* __descr = "copy into ";

    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator __first, Iterator __last) const
    {
        auto __src = make_constant_source<T>();
        if constexpr (__zipped)
        {
            // The zipped sequence holds a discard_iterator besides the sequence under test, so the source is zipped
            // with an arbitrary sequence to match its value type.
            auto __zip_src = oneapi::dpl::make_zip_iterator(__src, oneapi::dpl::counting_iterator<int>(0));
            oneapi::dpl::copy(std::forward<Policy>(exec), __zip_src, __zip_src + (__last - __first), __first);
        }
        else
        {
            oneapi::dpl::copy(std::forward<Policy>(exec), __src, __src + (__last - __first), __first);
        }
    }
};

// transform_iterator and permutation_iterator may leave elements of their base sequence unwritten, as the wrapper
// above does. A pattern requesting no_init for such a wrapped output must therefore not have that no_init reach the
// base sequence: discarding its content would destroy the elements which are not written.
//
// The sequence under test is a sycl::buffer constructed over host memory, and its content is only ever written from the
// host. That is what makes a discarding access mode observable: it elides the copy of the host data to the device, so
// the elements which the kernel does not write come back as garbage rather than as the values written on the host.
//
// __num_written is the number of leading elements the wrapped sequence is expected to write through to the base.
template <typename T, std::size_t __num_written, typename Policy, typename WrapIterator, typename RunPattern>
void
check_unwritten_elements_preserved(Policy&& exec, std::size_t n, std::size_t guard_size, T sentinel, WrapIterator wrap,
                                   RunPattern run, const std::string& descr)
{
    const std::size_t total_size = n + guard_size;

    std::vector<T> host_data(total_size, sentinel);
    for (std::size_t i = 0; i != n; ++i)
        host_data[i] = static_cast<T>(i);

    std::vector<T> expected(host_data);
    for (std::size_t i = 0; i != __num_written; ++i)
        expected[i] = new_value<T>;

    {
        sycl::buffer<T> buf(host_data.data(), total_size);
        auto __first = wrap(oneapi::dpl::begin(buf));
        run(std::forward<Policy>(exec), __first, __first + n);
    }
    // The buffer is destroyed, so its content has been written back into host_data.

    std::string msg = std::string("data destroyed by ") + RunPattern::__descr + descr;
    EXPECT_EQ_N(expected.begin(), host_data.begin(), total_size, msg.c_str());
}

// Runs the pattern over a single wrapper stack.
template <typename T, std::size_t __num_written, bool __zipped, typename Policy, typename WrapIterator>
void
check_stack(Policy&& exec, std::size_t n, std::size_t guard_size, T sentinel, WrapIterator wrap,
            const std::string& descr)
{
    check_unwritten_elements_preserved<T, __num_written>(std::forward<Policy>(exec), n, guard_size, sentinel, wrap,
                                                         run_copy<T, __zipped>{}, descr);
}

template <typename T, typename Policy>
void
call_check_unwritten_elements_preserved(Policy&& exec, std::size_t n, const std::string& type_text)
{
    if (!TestUtils::has_types_support<T>(exec.queue().get_device()))
    {
        TestUtils::unsupported_types_notifier(exec.queue().get_device());
        return;
    }
    constexpr std::size_t guard_size = 5;
    const T sentinel = static_cast<T>(-999);
    const std::string base_descr = std::string("(sycl_iterator<") + type_text + std::string(">)");
    oneapi::dpl::discard_iterator discard{};

    // No element of the base sequence is written.
    check_stack<T, /*__num_written=*/0, /*__zipped=*/false>(
        CLONE_TEST_POLICY_IDX(exec, 0), n, guard_size, sentinel,
        [](auto __it) { return oneapi::dpl::make_transform_iterator(__it, drop_write); },
        std::string("transform_iterator") + base_descr);

    // Only the 0th element of the base sequence is written.
    check_stack<T, /*__num_written=*/1, /*__zipped=*/false>(
        CLONE_TEST_POLICY_IDX(exec, 1), n, guard_size, sentinel,
        [](auto __it) { return oneapi::dpl::make_permutation_iterator(__it, map_to_zero); },
        std::string("permutation_iterator") + base_descr);

    // Wrapped a second time, to check that the no_init drop is not lost in the recursion.
    check_stack<T, /*__num_written=*/0, /*__zipped=*/false>(
        CLONE_TEST_POLICY_IDX(exec, 2), n, guard_size, sentinel,
        [](auto __it) {
            return oneapi::dpl::make_transform_iterator(oneapi::dpl::make_permutation_iterator(__it, map_to_zero),
                                                        drop_write);
        },
        std::string("transform_iterator(permutation_iterator") + base_descr + std::string(")"));

    check_stack<T, /*__num_written=*/0, /*__zipped=*/false>(
        CLONE_TEST_POLICY_IDX(exec, 3), n, guard_size, sentinel,
        [](auto __it) {
            return oneapi::dpl::make_permutation_iterator(oneapi::dpl::make_transform_iterator(__it, drop_write),
                                                          map_to_zero);
        },
        std::string("permutation_iterator(transform_iterator") + base_descr + std::string(")"));

    // A zip_iterator does not itself drop no_init for the sequences it zips, it propagates whatever the wrapper above
    // it dictates. With a transform_iterator or a permutation_iterator above it, the drop must reach the zipped
    // sycl_iterator through it.
    // The outermost wrapper is the transform_iterator, so the visible element is not a tuple.
    check_stack<T, /*__num_written=*/0, /*__zipped=*/false>(
        CLONE_TEST_POLICY_IDX(exec, 4), n, guard_size, sentinel,
        [discard](auto __it) {
            return oneapi::dpl::make_transform_iterator(oneapi::dpl::make_zip_iterator(__it, discard), drop_write);
        },
        std::string("transform_iterator(zip_iterator") + base_descr + std::string(")"));

    // Only the 0th element of the zipped sequence, and so the 0th element of the base sequence, is written.
    check_stack<T, /*__num_written=*/1, /*__zipped=*/true>(
        CLONE_TEST_POLICY_IDX(exec, 5), n, guard_size, sentinel,
        [discard](auto __it) {
            return oneapi::dpl::make_permutation_iterator(oneapi::dpl::make_zip_iterator(__it, discard), map_to_zero);
        },
        std::string("permutation_iterator(zip_iterator") + base_descr + std::string(")"));
}

template <typename T, int __recurse, typename Policy>
void
call_wrap_recurse(Policy&& exec, T trash, size_t n, const std::string& type_text)
{
    if (TestUtils::has_types_support<T>(exec.queue().get_device()))
    {
        constexpr size_t guard_size = 5;
        const size_t total_size = n + guard_size;
        const T sentinel = static_cast<T>(-999); // Distinct from trash

        TestUtils::usm_data_transfer<sycl::usm::alloc::shared, T> copy_out(exec, total_size);
        oneapi::dpl::counting_iterator<int> counting(0);
        // sycl iterator over a buffer whose memory is owned by the SYCL runtime
        {
            sycl::buffer<T> buf(total_size);
            //test all modes / wrappers
            wrap_recurse<__recurse, 0, /*__read =*/true, /*__reset_read=*/true, /*__write=*/true,
                         /*__check_write=*/true, /*__usable_as_perm_map=*/true, /*__usable_as_perm_src=*/true,
                         /*__is_reversible=*/false>(
                CLONE_TEST_POLICY_IDX(exec, 0), oneapi::dpl::begin(buf), oneapi::dpl::begin(buf) + n, counting,
                copy_out.get_data(), oneapi::dpl::begin(buf), copy_out.get_data(), counting, trash,
                std::string("sycl_iterator<") + type_text + std::string(">"), guard_size, sentinel);
        }
        // sycl iterator over a buffer constructed from host memory. An access mode which discards the content of
        // the buffer is observable here, because it elides the copy of the host data to the device.
        {
            std::vector<T> host_data(total_size);
            sycl::buffer<T> buf(host_data.data(), total_size);
            wrap_recurse<__recurse, 0, /*__read =*/true, /*__reset_read=*/true, /*__write=*/true,
                         /*__check_write=*/true, /*__usable_as_perm_map=*/true, /*__usable_as_perm_src=*/true,
                         /*__is_reversible=*/false>(
                CLONE_TEST_POLICY_IDX(exec, 1), oneapi::dpl::begin(buf), oneapi::dpl::begin(buf) + n, counting,
                copy_out.get_data(), oneapi::dpl::begin(buf), copy_out.get_data(), counting, trash,
                std::string("sycl_iterator<") + type_text + std::string(">(host memory)"), guard_size, sentinel);
        }
    }
    else
    {
        TestUtils::unsupported_types_notifier(exec.queue().get_device());
    }
}

template <typename Policy>
void
test_impl(Policy&& exec)
{
    constexpr size_t n = 10;
    
    // baseline with no wrapping
    call_wrap_recurse<float, 0>(CLONE_TEST_POLICY_IDX(exec, 0), -666.0f, n, "float");
    call_wrap_recurse<double, 0>(CLONE_TEST_POLICY_IDX(exec, 1), -666.0, n, "double");
    call_wrap_recurse<std::uint64_t, 0>(CLONE_TEST_POLICY_IDX(exec, 2), 999, n, "uint64_t");

    // big recursion step: 1 and 2 layers of wrapping
    call_wrap_recurse<std::int32_t, 2>(CLONE_TEST_POLICY_IDX(exec, 3), -666, n, "int32_t");

    // no_init requested for an output which is wrapped by an adapter that may skip writes must not reach the base
    call_check_unwritten_elements_preserved<std::int32_t>(CLONE_TEST_POLICY_IDX(exec, 4), n, "int32_t");
}

#endif //TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    auto policy = TestUtils::get_dpcpp_test_policy();
    test_impl(policy);

#if TEST_CHECK_COMPILATION_WITH_DIFF_POLICY_VAL_CATEGORY
    TestUtils::check_compilation(policy, [](auto&& policy) { test_impl(std::forward<decltype(policy)>(policy)); });
#endif
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
