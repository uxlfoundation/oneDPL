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

#include "xp_tests.hpp"

// Fixture
template <typename T> class DistributedVectorAllTypes : public testing::Test {
public:
};

TYPED_TEST_SUITE(DistributedVectorAllTypes, AllTypes);

TYPED_TEST(DistributedVectorAllTypes, StaticAsserts) {
  TypeParam dv(10);
  static_assert(stdrng::random_access_range<decltype(dv.segments())>);
  static_assert(stdrng::random_access_range<decltype(dv.segments()[0])>);
  static_assert(stdrng::viewable_range<decltype(dv.segments())>);

  static_assert(std::forward_iterator<decltype(dv.begin())>);
  static_assert(dr::distributed_iterator<decltype(dv.begin())>);

  static_assert(stdrng::forward_range<decltype(dv)>);
  static_assert(stdrng::random_access_range<decltype(dv)>);
  static_assert(dr::distributed_contiguous_range<decltype(dv)>);
}

TYPED_TEST(DistributedVectorAllTypes, getAndPut) {
  TypeParam dv(10);

  if (comm_rank == 0) {
    dv[5] = 13;
  } else {
  }
  fence_on(dv);

  for (std::size_t idx = 0; idx < 10; ++idx) {
    auto val = dv[idx];
    if (idx == 5) {
      EXPECT_EQ(val, 13);
    } else {
      EXPECT_NE(val, 13);
    }
  }
}

TYPED_TEST(DistributedVectorAllTypes, Stream) {
  Ops1<TypeParam> ops(10);
  std::ostringstream os;
  os << ops.dist_vec;
  EXPECT_EQ(os.str(), "{ 100, 101, 102, 103, 104, 105, 106, 107, 108, 109 }");
}

TYPED_TEST(DistributedVectorAllTypes, Equality) {
  Ops1<TypeParam> ops(10);
  iota(ops.dist_vec, 100);
  stdrng::iota(ops.vec, 100);
  EXPECT_TRUE(ops.dist_vec == ops.vec);
  EXPECT_EQ(ops.vec, ops.dist_vec);
}

TYPED_TEST(DistributedVectorAllTypes, Segments) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_segments(ops.dist_vec));
  EXPECT_TRUE(check_segments(stdrng::begin(ops.dist_vec)));
  EXPECT_TRUE(check_segments(stdrng::begin(ops.dist_vec) + 5));
}

TEST(DistributedVector, ConstructorBasic) {
  xp::distributed_vector<int> dist_vec(10);
  iota(dist_vec, 100);

  std::vector<int> local_vec(10);
  stdrng::iota(local_vec, 100);

  EXPECT_EQ(local_vec, dist_vec);
}

TEST(DistributedVector, ConstructorFill) {
  xp::distributed_vector<int> dist_vec(10, 1);

  std::vector<int> local_vec(10, 1);

  EXPECT_EQ(local_vec, dist_vec);
}

#ifndef DRISHMEM
TEST(DistributedVector, ConstructorBasicAOS) {
  OpsAOS ops(10);
  EXPECT_EQ(ops.vec, ops.dist_vec);
}

TEST(DistributedVector, ConstructorFillAOS) {
  AOS_Struct fill_value{1, 2};
  OpsAOS::dist_vec_type dist_vec(10, fill_value);
  OpsAOS::vec_type local_vec(10, fill_value);

  EXPECT_EQ(local_vec, dist_vec);
}
#endif