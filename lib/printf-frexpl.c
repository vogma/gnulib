/* Split a 'long double' into fraction and mantissa, for hexadecimal printf.
   Copyright (C) 2007, 2009-2025 Free Software Foundation, Inc.

   This file is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   This file is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

#include <config.h>

#if HAVE_SAME_LONG_DOUBLE_AS_DOUBLE

/* Specification.  */
# include "printf-frexpl.h"

# include "printf-frexp.h"

long double
printf_frexpl (long double x, int *expptr)
{
  return printf_frexp (x, expptr);
}

#else

# define USE_LONG_DOUBLE
# include "printf-frexp.c"

#endif
