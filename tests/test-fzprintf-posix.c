/* Test of POSIX compatible fzprintf() function.
   Copyright (C) 2007-2025 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* Written by Bruno Haible <bruno@clisp.org>, 2007.  */

#include <config.h>

#include <stdio.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "macros.h"

#define RETTYPE off64_t
#include "test-fprintf-posix.h"

int
main (int argc, char *argv[])
{
  test_function (fzprintf);
  return test_exit_status;
}
