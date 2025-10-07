// -*- C++ -*-
//===-----------------------------------------------------------------------===//
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

#ifndef _UTILS_SEQUENCE_H
#define _UTILS_SEQUENCE_H

// File contains common utilities that tests rely on

#include <iostream>
#include <iterator>
#include <vector>

#include "iterator_utils.h"

// Please uncomment this define if required to print full content of sequence.
// Otherwise only first 100 sequence items will be printed.
//#define PRINT_FULL_SEQUENCE_CONTENT 1

namespace TestUtils
{
// Sequence<T> is a container of a sequence of T with lots of kinds of iterators.
// Prefixes on begin/end mean:
//      c = "const"
//      f = "forward"
// No prefix indicates non-const random-access iterator.
template <typename T>
class Sequence
{
    ::std::vector<T> m_storage;

public:

    using value_type = T;

    using iterator       = typename ::std::vector<T>::iterator;
    using const_iterator = typename ::std::vector<T>::const_iterator;

    using forward_iterator       = ForwardIterator<iterator, ::std::forward_iterator_tag>;
    using const_forward_iterator = ForwardIterator<const_iterator, ::std::forward_iterator_tag>;

    using bidirectional_iterator       = BidirectionalIterator<iterator, ::std::bidirectional_iterator_tag>;
    using const_bidirectional_iterator = BidirectionalIterator<const_iterator, ::std::bidirectional_iterator_tag>;

    explicit Sequence(size_t size);

    // Construct sequence [f(0), f(1), ... f(size-1)]
    // f can rely on its invocations being sequential from 0 to size-1.
    template <typename Func>
    Sequence(size_t size, Func f);
    Sequence(const ::std::initializer_list<T>& data);

    const_iterator               begin   () const { return m_storage.begin();                                };
    const_iterator               end     () const { return m_storage.end();                                  };
    iterator                     begin   ()       { return m_storage.begin();                                };
    iterator                     end     ()       { return m_storage.end();                                  };
    const_iterator               cbegin  () const { return m_storage.cbegin();                               };
    const_iterator               cend    () const { return m_storage.cend();                                 };
    forward_iterator             fbegin  ()       { return forward_iterator(m_storage.begin());              };
    forward_iterator             fend    ()       { return forward_iterator(m_storage.end());                };
    const_forward_iterator       cfbegin () const { return const_forward_iterator(m_storage.cbegin());       };
    const_forward_iterator       cfend   () const { return const_forward_iterator(m_storage.cend());         };
    const_forward_iterator       fbegin  () const { return const_forward_iterator(m_storage.cbegin());       };
    const_forward_iterator       fend    () const { return const_forward_iterator(m_storage.cend());         };
    const_bidirectional_iterator cbibegin() const { return const_bidirectional_iterator(m_storage.cbegin()); };
    const_bidirectional_iterator cbiend  () const { return const_bidirectional_iterator(m_storage.cend());   };
    bidirectional_iterator       bibegin ()       { return bidirectional_iterator(m_storage.begin());        };
    bidirectional_iterator       biend   ()       { return bidirectional_iterator(m_storage.end());          };

    ::std::size_t size() const;
    T* data();
    const T* data() const;

    typename ::std::vector<T>::reference operator[](size_t j);
    typename ::std::vector<T>::const_reference operator[](size_t j) const;

    // Fill with given value
    void fill(const T& value);

    template <typename Func>
    void fill(Func f);

    void print() const;
};

//--------------------------------------------------------------------------------------------------------------------//
template <typename Iterator, typename F>
void
fill_data(Iterator first, Iterator last, F f)
{
    using T = typename std::iterator_traits<Iterator>::value_type;
    for (::std::size_t i = 0; first != last; (void) ++first, ++i)
    {
        *first = T(f(i));
    }
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename T>
Sequence<T>::Sequence(size_t size)
    : m_storage(size)
{
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename T>
template <typename Func>
Sequence<T>::Sequence(size_t size, Func f)
{
    m_storage.reserve(size);
    // Use push_back because T might not have a default constructor
    for (size_t k = 0; k < size; ++k)
        m_storage.push_back(T(f(k)));
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename T>
Sequence<T>::Sequence(const ::std::initializer_list<T>& data)
    : m_storage(data)
{
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename T>
::std::size_t
Sequence<T>::size() const
{
    return m_storage.size();
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename T>
T*
Sequence<T>::data()
{
    return m_storage.data();
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename T>
const T*
Sequence<T>::data() const
{
    return m_storage.data();
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename T>
typename ::std::vector<T>::reference
Sequence<T>::operator[](size_t j)
{
    return m_storage[j];
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename T>
typename ::std::vector<T>::const_reference
Sequence<T>::operator[](size_t j) const
{
    return m_storage[j];
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename T>
void
Sequence<T>::fill(const T& value)
{
    for (size_t i = 0; i < m_storage.size(); i++)
        m_storage[i] = value;
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename T>
template <typename Func>
void
Sequence<T>::fill(Func f)
{
    fill_data(m_storage.begin(), m_storage.end(), f);
}

//--------------------------------------------------------------------------------------------------------------------//
template <typename T>
void
Sequence<T>::print() const
{
    constexpr ::std::size_t max_print_count = 100;

    ::std::cout << "size = " << size() << ": { ";
#if PRINT_FULL_SEQUENCE_CONTENT
    ::std::copy(begin(), end(), ::std::ostream_iterator<T>(::std::cout, " "));
#else
    const auto printable_size = ::std::min(max_print_count, size());
    ::std::copy(begin(), begin() + printable_size, ::std::ostream_iterator<T>(::std::cout, " "));
#endif // PRINT_FULL_SEQUENCE_CONTENT
    ::std::cout << " } " << ::std::endl;
}

//--------------------------------------------------------------------------------------------------------------------//

} /* namespace TestUtils */

#endif // _UTILS_SEQUENCE_H
