// Copyright (C) 2015-2022 Free Software Foundation, Inc.
//
// This file is part of the GNU ISO C++ Library.  This library is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 2, or (at your option)
// any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this library; see the file COPYING3.  If not see
// <http://www.gnu.org/licenses/>.

//  specfun_testcase.h.h

//
//  These are little PODs for special function inputs and
//  expexted results for the testsuite.
//

#ifndef _TEST_SPECFUN_TESTCASE_H
#define _TEST_SPECFUN_TESTCASE_H

//  Generic cylindrical Bessel functions.
template <typename _Tp>
struct testcase_cyl_bessel
{
    _Tp f0;
    _Tp nu;
    _Tp x;
    _Tp f;
};

//  Regular modified cylindrical Bessel functions.
template <typename _Tp>
struct testcase_cyl_bessel_i
{
    _Tp f0;
    _Tp nu;
    _Tp x;
    _Tp f;
};

//  Cylindrical Bessel functions (of the first kind).
template <typename _Tp>
struct testcase_cyl_bessel_j
{
    _Tp f0;
    _Tp nu;
    _Tp x;
    _Tp f;
};

//  Irregular modified cylindrical Bessel functions.
template <typename _Tp>
struct testcase_cyl_bessel_k
{
    _Tp f0;
    _Tp nu;
    _Tp x;
    _Tp f;
};

#endif // _TEST_SPECFUN_TESTCASE_H
