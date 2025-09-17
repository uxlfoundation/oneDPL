// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#ifndef _COMPILE_ONLY_CHECKS_H
#define _COMPILE_ONLY_CHECKS_H

#include <iterator>    // std::iterator_traits
#include <type_traits> // std::decay_t, std::void_t, std::false_type, std::true_type
#include <utility>     // std::forward, std::move

#if TEST_DPCPP_BACKEND_PRESENT
#    include "oneapi/dpl/pstl/hetero/dpcpp/utils_ranges_sycl.h"
#endif

#include "iterator_utils.h"

namespace TestUtils
{

// Iterator adapter that deletes the comma operator to test comma operator protection
template <typename Iterator>
class NoCommaIterator
{
  public:
    using iterator_category = typename std::iterator_traits<Iterator>::iterator_category;
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using difference_type = typename std::iterator_traits<Iterator>::difference_type;
    using pointer = typename std::iterator_traits<Iterator>::pointer;
    using reference = typename std::iterator_traits<Iterator>::reference;

  private:
    Iterator iter_;

  public:
    // Constructors
    NoCommaIterator() = default;
    explicit NoCommaIterator(Iterator iter) : iter_(iter) {}

    // Copy and move constructors/assignment
    NoCommaIterator(const NoCommaIterator&) = default;
    NoCommaIterator(NoCommaIterator&&) = default;
    NoCommaIterator&
    operator=(const NoCommaIterator&) = default;
    NoCommaIterator&
    operator=(NoCommaIterator&&) = default;

    // Access to underlying iterator
    Iterator
    base() const
    {
        return iter_;
    }

    // Dereference operators
    reference
    operator*() const
    {
        return *iter_;
    }
    pointer
    operator->() const
    {
        return iter_.operator->();
    }
    reference
    operator[](difference_type n) const
    {
        return iter_[n];
    }

    // Increment/decrement operators
    NoCommaIterator&
    operator++()
    {
        ++iter_;
        return *this;
    }
    NoCommaIterator
    operator++(int)
    {
        NoCommaIterator tmp(*this);
        ++iter_;
        return tmp;
    }
    NoCommaIterator&
    operator--()
    {
        --iter_;
        return *this;
    }
    NoCommaIterator
    operator--(int)
    {
        NoCommaIterator tmp(*this);
        --iter_;
        return tmp;
    }

    // Arithmetic operators
    NoCommaIterator&
    operator+=(difference_type n)
    {
        iter_ += n;
        return *this;
    }
    NoCommaIterator&
    operator-=(difference_type n)
    {
        iter_ -= n;
        return *this;
    }
    NoCommaIterator
    operator+(difference_type n) const
    {
        return NoCommaIterator(iter_ + n);
    }
    NoCommaIterator
    operator-(difference_type n) const
    {
        return NoCommaIterator(iter_ - n);
    }
    difference_type
    operator-(const NoCommaIterator& other) const
    {
        return iter_ - other.iter_;
    }

    // Comparison operators
    bool
    operator==(const NoCommaIterator& other) const
    {
        return iter_ == other.iter_;
    }
    bool
    operator!=(const NoCommaIterator& other) const
    {
        return iter_ != other.iter_;
    }
    bool
    operator<(const NoCommaIterator& other) const
    {
        return iter_ < other.iter_;
    }
    bool
    operator<=(const NoCommaIterator& other) const
    {
        return iter_ <= other.iter_;
    }
    bool
    operator>(const NoCommaIterator& other) const
    {
        return iter_ > other.iter_;
    }
    bool
    operator>=(const NoCommaIterator& other) const
    {
        return iter_ >= other.iter_;
    }

    // Deleted comma operator - this is the key feature
    template<typename T>
    void operator,(T&&) = delete;
};

// Non-member arithmetic operators
template <typename Iterator>
NoCommaIterator<Iterator>
operator+(typename NoCommaIterator<Iterator>::difference_type n, const NoCommaIterator<Iterator>& iter)
{
    return iter + n;
}

// Helper function to create NoCommaIterator
template <typename Iterator>
NoCommaIterator<Iterator>
make_no_comma_iterator(Iterator iter)
{
    return NoCommaIterator<Iterator>(iter);
}
template <typename _T, typename = void>
struct __is_iterator_type : std::false_type
{
};

template <typename _T>
struct __is_iterator_type<_T, std::void_t<typename std::iterator_traits<_T>::difference_type>> : std::true_type
{
};

template <typename _T>
static constexpr bool __is_iterator_type_v = __is_iterator_type<_T>::value;

// Helper to conditionally wrap iterators with NoCommaIterator
template <typename T>
constexpr decltype(auto)
wrap_no_comma_if_iterator(T&& arg)
{
    if constexpr (__is_iterator_type_v<std::decay_t<T>>)
    {

#if TEST_DPCPP_BACKEND_PRESENT
        // avoid wrapping iterator-like buffer wrappers, or elements which must be transformed by our
        // data preparation before passing to sycl, as adding an iterator adapter around them causes problems.
        if constexpr (!oneapi::dpl::__ranges::__is_passed_directly_device_ready_v<std::decay_t<T>>)
        {
            return std::forward<T>(arg);
        }
        else
#endif
        {
            return make_no_comma_iterator(std::forward<T>(arg));
        }
    }
    else
        return std::forward<T>(arg);
}

template <typename Func>
struct callable_conv_to_no_comma_iters : std::decay_t<Func>
{
    using base_t = std::decay_t<Func>;
    callable_conv_to_no_comma_iters(base_t f) : base_t(f) {}

    template <typename... Args>
    void
    operator()(Args&&... args)
    {
        base_t::operator()(wrap_no_comma_if_iterator(std::forward<Args>(args))...);
    }
};

template <typename Policy, typename Op, typename... Args>
void
check_compilation_no_comma(Policy&& policy, Op&& op, Args&&... rest)
{
    //for libc++, we disable these checks because their sort implementation is broken for deleted comma operator iter
#if TEST_NO_COMMA_ITERATORS
    volatile bool always_false = false;
    if (always_false)
    {
        callable_conv_to_no_comma_iters<Op> wrapped_iter_op{std::forward<Op>(op)};
        iterator_invoker<std::random_access_iterator_tag, /*IsReverse*/ std::false_type>()(
            std::forward<Policy>(policy), wrapped_iter_op, std::forward<Args>(rest)...);
    }
#endif
}

template <typename _ExecutionPolicy>
struct compile_checker
{
    using _ExecutionPolicyDecayed = std::decay_t<_ExecutionPolicy>;

    _ExecutionPolicyDecayed my_policy;

    compile_checker(const _ExecutionPolicy& my_policy) : my_policy(my_policy) {}

    // Check compilation of callable argument with different policy value categories:
    // - compile for const _ExecutionPolicyDecayed&
    // - compile for _ExecutionPolicyDecayed&&
    template <typename _CallableTest>
    void
    compile(_CallableTest&& __callable_test)
    {
        // The goal of this check is to compile the same Kernel code with different policy type qualifiers.
        // This gives us ability to check that Kernel names generated inside oneDPL code are unique.
        volatile bool always_false = false;
        if (always_false)
        {
            // We just need to compile some Kernel code and we don't need to run this code in run-time
            // so we can move the rest of params again

            // Compile for const ExecutionPolicy&
            const auto& my_policy_ref = my_policy;
            __callable_test(my_policy_ref);

            // Compile for ExecutionPolicy&&
            __callable_test(std::move(my_policy));
        }
    }
};

// Check compilation of callable argument with different policy value categories:
// - compile for const _ExecutionPolicyDecayed&
// - compile for _ExecutionPolicyDecayed&&
template <typename _ExecutionPolicy, typename _CallableTest>
void
check_compilation(const _ExecutionPolicy& policy, _CallableTest&& __callable_test)
{
    compile_checker<_ExecutionPolicy>(policy).compile(std::forward<_CallableTest>(__callable_test));
}

} //namespace TestUtils

#endif // _COMPILE_ONLY_CHECKS_H
