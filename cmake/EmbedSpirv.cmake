# Copyright (c) 2026 Syluxium Industries.
# SPDX-License-Identifier: MIT

file(READ "${IN}" hex HEX)
string(LENGTH "${hex}" len)
set(words "")
math(EXPR last "${len} - 8")
foreach(i RANGE 0 ${last} 8)
    string(SUBSTRING "${hex}" ${i} 8 w)
    string(SUBSTRING "${w}" 6 2 b0)
    string(SUBSTRING "${w}" 4 2 b1)
    string(SUBSTRING "${w}" 2 2 b2)
    string(SUBSTRING "${w}" 0 2 b3)
    string(APPEND words "0x${b0}${b1}${b2}${b3}u,")
endforeach()
file(WRITE "${OUT}" "#pragma once\n#include <cstdint>\ninline constexpr uint32_t ${NAME}[] = {${words}};\n")
