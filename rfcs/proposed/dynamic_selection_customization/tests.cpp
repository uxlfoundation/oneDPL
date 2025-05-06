// icpx -fsycl -std=c++17 -o tests.exe tests.cpp -ltbb

// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/dynamic_selection>
#include <tbb/tbb.h>
#include <cstdio>
#include <utility>

namespace ex = oneapi::dpl::experimental;

using pair_t = std::pair<tbb::task_arena *, tbb::task_group *>;

void numa_1() {
    // create pairs of arenas and task_groups, one per numa node
    std::vector<tbb::numa_node_id> numa_nodes = tbb::info::numa_nodes();
    std::vector<pair_t> pairs;

    for (int i = 0; i < numa_nodes.size(); i++) {
        pairs.emplace_back(pair_t(new tbb::task_arena{tbb::task_arena::constraints(numa_nodes[i]), 0},
                                  new tbb::task_group{}) );
    }
    // end creating default arenas and groups

    ex::round_robin_policy<pair_t> rr{ pairs };

    // helper struct for waiting on the work in the pair
    struct WaitType {
        pair_t pair;
        void wait() { pair.first->execute([this]() { pair.second->wait(); }); }
    };
    std::vector<WaitType> submissions;

    for (auto i : numa_nodes) {
        auto w = ex::submit( rr, 
                             [](pair_t p) {
                                p.first->enqueue(p.second->defer([]() { std::printf("o\n"); }));
                                return WaitType{ p };
                             }
                            );
        submissions.emplace_back(w.unwrap());
    }

    for (auto& s : submissions)
        ex::wait(s);
}

namespace numa {

    class ArenaAndGroup {
        tbb::task_arena *a_;
        tbb::task_group *tg_;
    public:
        ArenaAndGroup(tbb::task_arena *a, tbb::task_group *tg) : a_(a), tg_(tg) {}
      
        template<typename F>
        auto run(F&& f) {
            a_->enqueue(tg_->defer([&]() { std::forward<F>(f)(); }));
            return *this;
        }

        void wait() { 
            a_->execute([this]() { tg_->wait(); }); 
        }

        void clear() { delete a_; delete tg_; }
    };

    class numa_backend : public ex::backend_base<ArenaAndGroup, numa_backend> {
    public:
        using resource_type = ArenaAndGroup;
        using my_base = backend_base<ArenaAndGroup, numa_backend>;
        numa_backend() : my_base(), owns_groups_(true) { 
            std::vector<tbb::numa_node_id> numa_nodes = tbb::info::numa_nodes();
            for (int i = 0; i < numa_nodes.size(); i++) {
                resources_.emplace_back( ArenaAndGroup(new tbb::task_arena{tbb::task_arena::constraints(numa_nodes[i]), 0},
                                                       new tbb::task_group{}) );
            }
        }

        numa_backend(const std::vector<ArenaAndGroup>& u) : my_base(u) {  }

        ~numa_backend() {
            if (owns_groups_)
                for (auto& r : resources_) 
                    r.clear();
        }

    private:
        bool owns_groups_ = false;
    };

}

void numa_2() {
    std::vector<tbb::numa_node_id> numa_nodes = tbb::info::numa_nodes();

    ex::round_robin_policy<numa::ArenaAndGroup, numa::numa_backend> rr{ };
    for (auto i : numa_nodes) {
        ex::submit(rr, 
            [](numa::ArenaAndGroup ag) { 
                ag.run([]() { std::printf("o\n"); });
                return ag; }
        );
    }
    ex::wait(rr.get_submission_group());
}

void no_wait_support() {
    tbb::task_group t1, t2;

    ex::round_robin_policy<tbb::task_group*> p{ { &t1, &t2 } };

    auto g = p.get_submission_group();
    try {
        ex::wait(g);
    } catch (std::logic_error& e) {
        std::cout << "Failed as expected: " << e.what() << "\n";
    }

    struct tpw {
        tbb::task_group *tg;
        void wait() { tg->wait(); }
    };

    ex::round_robin_policy<tpw> p2{ { tpw{&t1}, tpw{&t2} } };
    auto g2 = p2.get_submission_group();
    ex::wait(g2);
    std::printf("Ok\n");
}

int main() {
    no_wait_support();
    std::printf("---\n");
    numa_1();
    std::printf("---\n");
    numa_2();
    return 0;
}
