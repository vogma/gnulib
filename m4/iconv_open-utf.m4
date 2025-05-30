# iconv_open-utf.m4
# serial 6
dnl Copyright (C) 2007-2025 Free Software Foundation, Inc.
dnl This file is free software; the Free Software Foundation
dnl gives unlimited permission to copy and/or distribute it,
dnl with or without modifications, as long as this notice is preserved.
dnl This file is offered as-is, without any warranty.

# A placeholder to ensure that this m4 file gets included by aclocal.
AC_DEFUN([gl_FUNC_ICONV_OPEN_UTF], [])

AC_DEFUN([gl_FUNC_ICONV_OPEN_UTF_SUPPORT],
[
  dnl This macro relies on am_cv_func_iconv and gl_func_iconv_gnu from
  dnl gl_FUNC_ICONV_OPEN, but is called from within gl_FUNC_ICONV_OPEN.
  dnl *Not* AC_REQUIRE([gl_FUNC_ICONV_OPEN]).
  AC_REQUIRE([AC_CANONICAL_HOST]) dnl for cross-compiles
  AC_REQUIRE([gl_ICONV_H_DEFAULTS])
  if test "$am_cv_func_iconv" = yes; then
    AC_CACHE_CHECK([whether iconv supports conversion between UTF-8 and UTF-{16,32}{BE,LE}],
      [gl_cv_func_iconv_supports_utf],
      [
        saved_LIBS="$LIBS"
        LIBS="$LIBS $LIBICONV"
        AC_RUN_IFELSE(
          [AC_LANG_SOURCE([[
#include <iconv.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main ()
{
  int result = 0;
  /* Test conversion from UTF-8 to UTF-16BE with no errors.  */
  {
    static const char input[] =
      "Japanese (\346\227\245\346\234\254\350\252\236) [\360\235\224\215\360\235\224\236\360\235\224\255]";
    static const char expected[] =
      "\000J\000a\000p\000a\000n\000e\000s\000e\000 \000(\145\345\147\054\212\236\000)\000 \000[\330\065\335\015\330\065\335\036\330\065\335\055\000]";
    iconv_t cd;
    cd = iconv_open ("UTF-16BE", "UTF-8");
    if (cd == (iconv_t)(-1))
      result |= 1;
    else
      {
        char buf[100];
        const char *inptr;
        size_t inbytesleft;
        char *outptr;
        size_t outbytesleft;
        size_t res;
        inptr = input;
        inbytesleft = sizeof (input) - 1;
        outptr = buf;
        outbytesleft = sizeof (buf);
        res = iconv (cd,
                     (ICONV_CONST char **) &inptr, &inbytesleft,
                     &outptr, &outbytesleft);
        if (!(res == 0 && inbytesleft == 0))
          result |= 1;
        else if (!(outptr == buf + (sizeof (expected) - 1)))
          result |= 1;
        else if (!(memcmp (buf, expected, sizeof (expected) - 1) == 0))
          result |= 1;
        else if (!(iconv_close (cd) == 0))
          result |= 1;
      }
  }
  /* Test conversion from UTF-8 to UTF-16LE with no errors.  */
  {
    static const char input[] =
      "Japanese (\346\227\245\346\234\254\350\252\236) [\360\235\224\215\360\235\224\236\360\235\224\255]";
    static const char expected[] =
      "J\000a\000p\000a\000n\000e\000s\000e\000 \000(\000\345\145\054\147\236\212)\000 \000[\000\065\330\015\335\065\330\036\335\065\330\055\335]\000";
    iconv_t cd;
    cd = iconv_open ("UTF-16LE", "UTF-8");
    if (cd == (iconv_t)(-1))
      result |= 2;
    else
      {
        char buf[100];
        const char *inptr;
        size_t inbytesleft;
        char *outptr;
        size_t outbytesleft;
        size_t res;
        inptr = input;
        inbytesleft = sizeof (input) - 1;
        outptr = buf;
        outbytesleft = sizeof (buf);
        res = iconv (cd,
                     (ICONV_CONST char **) &inptr, &inbytesleft,
                     &outptr, &outbytesleft);
        if (!(res == 0 && inbytesleft == 0))
          result |= 2;
        else if (!(outptr == buf + (sizeof (expected) - 1)))
          result |= 2;
        else if (!(memcmp (buf, expected, sizeof (expected) - 1) == 0))
          result |= 2;
        else if (!(iconv_close (cd) == 0))
          result |= 2;
      }
  }
  /* Test conversion from UTF-8 to UTF-32BE with no errors.  */
  {
    static const char input[] =
      "Japanese (\346\227\245\346\234\254\350\252\236) [\360\235\224\215\360\235\224\236\360\235\224\255]";
    static const char expected[] =
      "\000\000\000J\000\000\000a\000\000\000p\000\000\000a\000\000\000n\000\000\000e\000\000\000s\000\000\000e\000\000\000 \000\000\000(\000\000\145\345\000\000\147\054\000\000\212\236\000\000\000)\000\000\000 \000\000\000[\000\001\325\015\000\001\325\036\000\001\325\055\000\000\000]";
    iconv_t cd;
    cd = iconv_open ("UTF-32BE", "UTF-8");
    if (cd == (iconv_t)(-1))
      result |= 4;
    else
      {
        char buf[100];
        const char *inptr;
        size_t inbytesleft;
        char *outptr;
        size_t outbytesleft;
        size_t res;
        inptr = input;
        inbytesleft = sizeof (input) - 1;
        outptr = buf;
        outbytesleft = sizeof (buf);
        res = iconv (cd,
                     (ICONV_CONST char **) &inptr, &inbytesleft,
                     &outptr, &outbytesleft);
        if (!(res == 0 && inbytesleft == 0))
          result |= 4;
        else if (!(outptr == buf + (sizeof (expected) - 1)))
          result |= 4;
        else if (!(memcmp (buf, expected, sizeof (expected) - 1) == 0))
          result |= 4;
        else if (!(iconv_close (cd) == 0))
          result |= 4;
      }
  }
  /* Test conversion from UTF-8 to UTF-32LE with no errors.  */
  {
    static const char input[] =
      "Japanese (\346\227\245\346\234\254\350\252\236) [\360\235\224\215\360\235\224\236\360\235\224\255]";
    static const char expected[] =
      "J\000\000\000a\000\000\000p\000\000\000a\000\000\000n\000\000\000e\000\000\000s\000\000\000e\000\000\000 \000\000\000(\000\000\000\345\145\000\000\054\147\000\000\236\212\000\000)\000\000\000 \000\000\000[\000\000\000\015\325\001\000\036\325\001\000\055\325\001\000]\000\000\000";
    iconv_t cd;
    cd = iconv_open ("UTF-32LE", "UTF-8");
    if (cd == (iconv_t)(-1))
      result |= 8;
    else
      {
        char buf[100];
        const char *inptr;
        size_t inbytesleft;
        char *outptr;
        size_t outbytesleft;
        size_t res;
        inptr = input;
        inbytesleft = sizeof (input) - 1;
        outptr = buf;
        outbytesleft = sizeof (buf);
        res = iconv (cd,
                     (ICONV_CONST char **) &inptr, &inbytesleft,
                     &outptr, &outbytesleft);
        if (!(res == 0 && inbytesleft == 0))
          result |= 8;
        else if (!(outptr == buf + (sizeof (expected) - 1)))
          result |= 8;
        else if (!(memcmp (buf, expected, sizeof (expected) - 1) == 0))
          result |= 8;
        else if (!(iconv_close (cd) == 0))
          result |= 8;
      }
  }
  /* Test conversion from UTF-16BE to UTF-8 with no errors.
     This test fails on NetBSD 3.0.  */
  {
    static const char input[] =
      "\000J\000a\000p\000a\000n\000e\000s\000e\000 \000(\145\345\147\054\212\236\000)\000 \000[\330\065\335\015\330\065\335\036\330\065\335\055\000]";
    static const char expected[] =
      "Japanese (\346\227\245\346\234\254\350\252\236) [\360\235\224\215\360\235\224\236\360\235\224\255]";
    iconv_t cd;
    cd = iconv_open ("UTF-8", "UTF-16BE");
    if (cd == (iconv_t)(-1))
      result |= 16;
    else
      {
        char buf[100];
        const char *inptr;
        size_t inbytesleft;
        char *outptr;
        size_t outbytesleft;
        size_t res;
        inptr = input;
        inbytesleft = sizeof (input) - 1;
        outptr = buf;
        outbytesleft = sizeof (buf);
        res = iconv (cd,
                     (ICONV_CONST char **) &inptr, &inbytesleft,
                     &outptr, &outbytesleft);
        if (!(res == 0 && inbytesleft == 0))
          result |= 16;
        else if (!(outptr == buf + (sizeof (expected) - 1)))
          result |= 16;
        else if (!(memcmp (buf, expected, sizeof (expected) - 1) == 0))
          result |= 16;
        else if (!(iconv_close (cd) == 0))
          result |= 16;
      }
  }
  return result;
}]])],
          [gl_cv_func_iconv_supports_utf=yes],
          [gl_cv_func_iconv_supports_utf=no],
          [
           dnl We know that GNU libiconv, GNU libc, musl libc, and Solaris >= 9 do.
           dnl OSF/1 5.1 has these encodings, but inserts a BOM in the "to"
           dnl direction.
           gl_cv_func_iconv_supports_utf="$gl_cross_guess_normal"
           if test $gl_func_iconv_gnu = yes; then
             gl_cv_func_iconv_supports_utf="guessing yes"
           else
changequote(,)dnl
             case "$host_os" in
               *-musl* | midipix*)           gl_cv_func_iconv_supports_utf="guessing yes" ;;
               solaris2.9 | solaris2.1[0-9]) gl_cv_func_iconv_supports_utf="guessing yes" ;;
             esac
changequote([,])dnl
           fi
          ])
        LIBS="$saved_LIBS"
      ])
  fi
])
