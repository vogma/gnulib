/* Test of execlp().
   Copyright (C) 2020-2025 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

/* Written by Bruno Haible <bruno@clisp.org>, 2020.  */

#include <config.h>

/* Specification.  */
#include <unistd.h>

#include "signature.h"
SIGNATURE_CHECK (execlp, int, (const char *, const char *, ...));

#include <stdio.h>

int
main ()
{
  const char *progname = "test-exec-child";
  execlp (progname,
          progname,
          "abc def",
          "abc\"def\"ghi",
          "xyz\"",
          "abc\\def\\ghi",
          "xyz\\",
          "???",
          "***",
          "",
          "foo",
          "",
          NULL);

  perror ("execlp");
  return 1;
}
