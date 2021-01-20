#ifndef SOKUDO_COMMON_H
#define SOKUDO_COMMON_H
#include "errors.h"

inline void sokudo_assert(bool exp, const std::string &msg = "Assertion failed") {
    if (!exp) {
        throw sokudo::errors::InvalidOperationException(msg);
    }
}

#endif
