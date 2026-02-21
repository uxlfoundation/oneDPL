// -*- C++ -*-
//===---------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_BARRIERS_H
#define _ONEDPL_BARRIERS_H


#include <thread>
#include <mutex>
#include <condition_variable>

class Barrier {
public:
    explicit Barrier(std::size_t count) : threshold(count), count(count), generation(0) {}
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex);
        auto gen = generation;
        if (--count == 0) {
            generation++;
            count = threshold;
            condition.notify_all();
        } else {
            condition.wait(lock, [this, gen] { return gen != generation; });
        }
    }

private:
    std::mutex mutex;
    std::condition_variable condition;
    std::size_t threshold;
    std::size_t count;
    std::size_t generation;

};

#endif /* _ONEDPL_BARRIERS_H */
