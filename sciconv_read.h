/*
    This file is part of Sciscipy.

    Sciscipy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Sciscipy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
    
    Copyright (c) 2009, Vincent Guffens.
*/




#ifndef SCICONV_READ
#define SCICONV_READ

#include <Python.h>

#include "util.h"
#if NUMPY == 1
#include "numpy/arrayobject.h"
#endif

struct sciconv_read_struct 
{
	PyObject * (*conv_func)(int *) ;  	
	int scitype ;
	struct sciconv_read_struct *next ;
} ;

// List of converter functions 
extern struct sciconv_read_struct* sciconv_read_list ;

void sciconv_read_init(void) ;
// Generic read a scilab variable given its address and type
PyObject * sciconv_read (int *addr, int var_type) ;

#endif 
