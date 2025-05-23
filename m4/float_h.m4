# float_h.m4
# serial 15
dnl Copyright (C) 2007, 2009-2025 Free Software Foundation, Inc.
dnl This file is free software; the Free Software Foundation
dnl gives unlimited permission to copy and/or distribute it,
dnl with or without modifications, as long as this notice is preserved.
dnl This file is offered as-is, without any warranty.

AC_DEFUN([gl_FLOAT_H],
[
  AC_REQUIRE([AC_PROG_CC])
  AC_REQUIRE([AC_CANONICAL_HOST])
  GL_GENERATE_FLOAT_H=false
  REPLACE_FLOAT_LDBL=0
  case "$host_os" in
    aix* | beos* | openbsd* | mirbsd* | irix*)
      GL_GENERATE_FLOAT_H=true
      ;;
    freebsd* | dragonfly*)
      case "$host_cpu" in
changequote(,)dnl
        i[34567]86 )
changequote([,])dnl
          GL_GENERATE_FLOAT_H=true
          ;;
        x86_64 )
          # On x86_64 systems, the C compiler may still be generating
          # 32-bit code.
          AC_COMPILE_IFELSE(
            [AC_LANG_SOURCE(
               [[#if defined __LP64__ || defined __x86_64__ || defined __amd64__
                  int ok;
                 #else
                  error fail
                 #endif
               ]])],
            [],
            [GL_GENERATE_FLOAT_H=true])
          ;;
      esac
      ;;
    linux*)
      case "$host_cpu" in
        powerpc*)
          GL_GENERATE_FLOAT_H=true
          ;;
      esac
      ;;
  esac
  case "$host_os" in
    aix* | freebsd* | dragonfly* | linux*)
      if $GL_GENERATE_FLOAT_H; then
        REPLACE_FLOAT_LDBL=1
      fi
      ;;
  esac

  dnl Test for completeness w.r.t. ISO C 23.
  REPLACE_FLOAT_SNAN=0
  AC_CACHE_CHECK([whether float.h conforms to ISO C23],
    [gl_cv_header_float_h_isoc23],
    [AC_COMPILE_IFELSE(
       [AC_LANG_PROGRAM(
          [[#include <float.h>
            int x[] = { FLT_DECIMAL_DIG, DBL_DECIMAL_DIG, LDBL_DECIMAL_DIG };
            float maxf = FLT_NORM_MAX;
            double maxd = DBL_NORM_MAX;
            long double maxl = LDBL_NORM_MAX;
          ]],
          [[float sf = FLT_SNAN;
            double sd = DBL_SNAN;
            long double sl = LDBL_SNAN;
            return (sf != 0) + (sd != 0) + (sl != 0);
          ]])],
       [gl_cv_header_float_h_isoc23=yes],
       [gl_cv_header_float_h_isoc23=no])
    ])
  if test $gl_cv_header_float_h_isoc23 != yes; then
    GL_GENERATE_FLOAT_H=true
    REPLACE_FLOAT_SNAN=1
  fi

  dnl Test against glibc-2.7 Linux/SPARC64 bug.
  REPLACE_ITOLD=0
  AC_CACHE_CHECK([whether conversion from 'int' to 'long double' works],
    [gl_cv_func_itold_works],
    [
      AC_RUN_IFELSE(
        [AC_LANG_SOURCE([[
int i = -1;
volatile long double ld;
int main ()
{
  ld += i * 1.0L;
  if (ld > 0)
    return 1;
  return 0;
}]])],
        [gl_cv_func_itold_works=yes],
        [gl_cv_func_itold_works=no],
        [case "$host" in
           sparc*-*-linux*)
             AC_COMPILE_IFELSE(
               [AC_LANG_SOURCE(
                 [[#if defined __LP64__ || defined __arch64__
                    int ok;
                   #else
                    error fail
                   #endif
                 ]])],
               [gl_cv_func_itold_works="guessing no"],
               [gl_cv_func_itold_works="guessing yes"])
             ;;
             # Guess yes on native Windows.
           mingw* | windows*)
             gl_cv_func_itold_works="guessing yes" ;;
           *)
             gl_cv_func_itold_works="guessing yes" ;;
         esac
        ])
    ])
  case "$gl_cv_func_itold_works" in
    *no)
      REPLACE_ITOLD=1
      dnl We add the workaround to <float.h> but also to <math.h>,
      dnl to increase the chances that the fix function gets pulled in.
      GL_GENERATE_FLOAT_H=true
      ;;
  esac

  if $GL_GENERATE_FLOAT_H; then
    gl_NEXT_HEADERS([float.h])
  fi
  AC_SUBST([REPLACE_ITOLD])
])
