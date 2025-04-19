# crc-x86_64.m4
# serial 3
dnl Copyright (C) 2024-2025 Free Software Foundation, Inc.
dnl This file is free software; the Free Software Foundation
dnl gives unlimited permission to copy and/or distribute it,
dnl with or without modifications, as long as this notice is preserved.
dnl This file is offered as-is, without any warranty.

AC_DEFUN([gl_CRC_AARCH64_PMULL],
[
  AC_CACHE_CHECK([if pmull intrinsic exists], [gl_cv_crc_pmull], [
    AC_LINK_IFELSE(
      [AC_LANG_SOURCE(
        [[
          #include <arm_neon.h>
          #include <arm_acle.h>

          #if defined __GNUC__ || defined __clang__
          __attribute__ ((__target__ ("+crc+crypto+sha3")))
          #endif
          int
          main (void)
          {
            uint32_t c = __crc32b(0, 0);

            poly64_t  p = (poly64_t)0;
            poly128_t r = vmull_p64(p, p);

            uint64x2_t a = vdupq_n_u64(0), b = vdupq_n_u64(1);
            uint64x2_t s = veor3q_u64(a, b, a);

            return 0;
          }
        ]])
      ], [
        gl_cv_crc_pmull=yes
      ], [
        gl_cv_crc_pmull=no
      ])
  ])
  if test $gl_cv_crc_pmull = yes; then
    AC_DEFINE([GL_CRC_AARCH64_PMULL], [1],
              [CRC32 calculation by pmull hardware instruction enabled])
  fi
  AM_CONDITIONAL([GL_CRC_AARCH64_PMULL],
                 [test $gl_cv_crc_pmull = yes])
])
