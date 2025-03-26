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
#ifndef _UTILS_DEVICE_COPYABLE_H
#define _UTILS_DEVICE_COPYABLE_H

#if TEST_DPCPP_BACKEND_PRESENT
#include "utils_sycl_defs.h"
#include <iostream>
#include <type_traits>
#include <cstddef>
#include <iterator>

namespace TestUtils
{

// Device copyable noop functor used in testing as surrogate for predicates, binary ops, unary functors
// Intentionally non-trivially copyable to test that device_copyable specialization works and we are not
// relying on trivial copyability
struct noop_device_copyable
{
    noop_device_copyable(const noop_device_copyable& other) { std::cout << "non trivial copy ctor\n"; }
    int
    operator()(int a) const
    {
        return a;
    }
};

struct noop_non_device_copyable
{
    noop_non_device_copyable(const noop_non_device_copyable& other) { std::cout << "non trivial copy ctor\n"; }
    int
    operator()(int a) const
    {
        return a;
    }
};

// Device copyable assignment callable.
// Intentionally non-trivially copyable to test that device_copyable specialization works and we are not
// relying on trivial copyability
struct assign_non_device_copyable
{
    assign_non_device_copyable(const assign_non_device_copyable& other) { std::cout << "non trivial copy ctor\n"; }
    template <typename _Xp, typename _Yp>
    void
    operator()(const _Xp& __x, _Yp&& __y) const
    {
        std::forward<_Yp>(__y) = __x;
    }
};

struct assign_device_copyable
{
    assign_device_copyable(const assign_device_copyable& other) { std::cout << "non trivial copy ctor\n"; }
    template <typename _Xp, typename _Yp>
    void
    operator()(const _Xp& __x, _Yp&& __y) const
    {
        std::forward<_Yp>(__y) = __x;
    }
};

// Device copyable binary operator binary operators.
// Intentionally non-trivially copyable to test that device_copyable specialization works and we are not
// relying on trivial copyability
struct binary_op_non_device_copyable
{
    binary_op_non_device_copyable(const binary_op_non_device_copyable& other)
    {
        std::cout << " non trivial copy ctor\n";
    }
    int
    operator()(int a, int b) const
    {
        return a;
    }
};

struct binary_op_device_copyable
{
    binary_op_device_copyable(const binary_op_device_copyable& other) { std::cout << " non trivial copy ctor\n"; }
    int
    operator()(int a, int b) const
    {
        return a;
    }
};

// Device copyable int wrapper struct used in testing as surrogate for values, value types, etc.
// Intentionally non-trivially copyable to test that device_copyable specialization works and we are not
// relying on trivial copyability
struct int_device_copyable
{
    int i;
    int_device_copyable(const int_device_copyable& other) : i(other.i) { std::cout << "non trivial copy ctor\n"; }
};

struct int_non_device_copyable
{
    int i;
    int_non_device_copyable(const int_non_device_copyable& other) : i(other.i)
    {
        std::cout << "non trivial copy ctor\n";
    }
};

// Device copyable iterator used in testing as surrogate for iterators.
// Intentionally non-trivially copyable to test that device_copyable specialization works and we are not
// relying on trivial copyability
struct constant_iterator_device_copyable
{
    using iterator_category = std::random_access_iterator_tag;
    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using pointer = const int*;
    using reference = const int&;

    int i;
    constant_iterator_device_copyable(int __i) : i(__i) {}

    constant_iterator_device_copyable(const constant_iterator_device_copyable& other) : i(other.i)
    {
        std::cout << "non trivial copy ctor\n";
    }

    reference operator*() const { return i; }

    constant_iterator_device_copyable& operator++() { return *this; }
    constant_iterator_device_copyable operator++(int) {return *this; }

    constant_iterator_device_copyable& operator--() {  return *this; }
    constant_iterator_device_copyable operator--(int) { return *this; }

    constant_iterator_device_copyable& operator+=(difference_type n) {return *this; }
    constant_iterator_device_copyable operator+(difference_type n) const { return constant_iterator_device_copyable(i); }
    friend constant_iterator_device_copyable operator+(difference_type n, const constant_iterator_device_copyable& it) { return constant_iterator_device_copyable(it.i); }

    constant_iterator_device_copyable& operator-=(difference_type n) { return *this; }
    constant_iterator_device_copyable operator-(difference_type n) const { return constant_iterator_device_copyable(i); }
    difference_type operator-(const constant_iterator_device_copyable& other) const { return 0; }

    reference operator[](difference_type n) const { return i; }

    bool operator==(const constant_iterator_device_copyable& other) const { return true; }
    bool operator!=(const constant_iterator_device_copyable& other) const { return false; }
    bool operator<(const constant_iterator_device_copyable& other) const { return false; }
    bool operator>(const constant_iterator_device_copyable& other) const { return false; }
    bool operator<=(const constant_iterator_device_copyable& other) const { return true; }
    bool operator>=(const constant_iterator_device_copyable& other) const { return true; }
};

struct constant_iterator_non_device_copyable
{
    using iterator_category = std::random_access_iterator_tag;
    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using pointer = const int*;
    using reference = const int&;

    int i;
    constant_iterator_non_device_copyable(int __i) : i(__i) {}

    constant_iterator_non_device_copyable(const constant_iterator_non_device_copyable& other) : i(other.i)
    {
        std::cout << "non trivial copy ctor\n";
    }

    reference operator*() const { return i; }

    constant_iterator_non_device_copyable& operator++() { return *this; }
    constant_iterator_non_device_copyable operator++(int) {return *this; }

    constant_iterator_non_device_copyable& operator--() {  return *this; }
    constant_iterator_non_device_copyable operator--(int) { return *this; }

    constant_iterator_non_device_copyable& operator+=(difference_type n) {return *this; }
    constant_iterator_non_device_copyable operator+(difference_type n) const { return constant_iterator_non_device_copyable(i); }
    friend constant_iterator_non_device_copyable operator+(difference_type n, const constant_iterator_non_device_copyable& it) { return constant_iterator_non_device_copyable(it.i); }

    constant_iterator_non_device_copyable& operator-=(difference_type n) { return *this; }
    constant_iterator_non_device_copyable operator-(difference_type n) const { return constant_iterator_non_device_copyable(i); }
    difference_type operator-(const constant_iterator_non_device_copyable& other) const { return 0; }

    reference operator[](difference_type n) const { return i; }

    bool operator==(const constant_iterator_non_device_copyable& other) const { return true; }
    bool operator!=(const constant_iterator_non_device_copyable& other) const { return false; }
    bool operator<(const constant_iterator_non_device_copyable& other) const { return false; }
    bool operator>(const constant_iterator_non_device_copyable& other) const { return false; }
    bool operator<=(const constant_iterator_non_device_copyable& other) const { return true; }
    bool operator>=(const constant_iterator_non_device_copyable& other) const { return true; }
};

// Non-trivially copyable ranges used in testing as surrogate for ranges.
// Intentionally non-trivially copyable to test that device_copyable specialization (range_device_copyable) works
// and we are not relying on trivial copyability
class range_non_device_copyable
{
public:
    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using pointer = int*;
    using reference = int&;

    pointer begin() const { return this->data(); }
    pointer end() const { return this->data() + this->size(); }
    pointer data() const { return m_data; }
    difference_type size() const { return m_size; }
    reference operator[](difference_type i) const { return m_data[i]; }

    range_non_device_copyable(const range_non_device_copyable& other) : m_data(other.data()), m_size(other.size())
    {
        std::cout << "non trivial copy ctor\n";
    }
    range_non_device_copyable(pointer data, difference_type size) : m_data(data), m_size(size) {}
private:
    pointer m_data = nullptr;
    difference_type m_size = 0;
};
class range_device_copyable: public range_non_device_copyable {};

} /* namespace TestUtils */

template <>
struct sycl::is_device_copyable<TestUtils::noop_device_copyable> : std::true_type
{
};

template <>
struct sycl::is_device_copyable<TestUtils::assign_device_copyable> : std::true_type
{
};

template <>
struct sycl::is_device_copyable<TestUtils::binary_op_device_copyable> : std::true_type
{
};

template <>
struct sycl::is_device_copyable<TestUtils::int_device_copyable> : std::true_type
{
};

template <>
struct sycl::is_device_copyable<TestUtils::constant_iterator_device_copyable> : std::true_type
{
};

template <>
struct sycl::is_device_copyable<TestUtils::range_device_copyable> : std::true_type
{
};
#endif // TEST_DPCPP_BACKEND_PRESENT

#endif // _UTILS_DEVICE_COPYABLE_H
