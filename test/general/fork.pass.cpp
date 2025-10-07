// -*- C++ -*-
//===-- fork.pass.cpp -----------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/execution>
#include "support/utils.h"

#if __has_include(<unistd.h>) && __has_include(<sys/wait.h>)
#include <unistd.h>
#include <sys/wait.h>
#include <cerrno>
#endif

int main()
{
#ifdef _POSIX_VERSION // defined in <unistd.h>
    pid_t pid = fork();
    if (pid < 0) // error
        return errno;
    if (pid == 0) // child
        return 0;
    int status;
    pid_t pid2 = waitpid(pid, &status, 0);
    if (pid2 != pid) // error
        return errno;
    if (WIFEXITED(status))
        return WEXITSTATUS(status);
    return -1;
#else
    return TestUtils::done(0); // skipped
#endif
}
