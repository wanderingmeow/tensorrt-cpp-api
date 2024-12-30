#pragma once

#include <iostream>
#include <cstdlib>

#define CHECK(condition)                                                                                                                   \
    do {                                                                                                                                   \
        if (!(condition)) {                                                                                                                \
            std::cerr << "Assertion failed: (" << #condition << "), function " << __FUNCTION__ << ", file " << __FILE__ << ", line " << __LINE__ << ".\n"; \
            std::abort();                                                                                                                  \
        }                                                                                                                                  \
    } while (false);
