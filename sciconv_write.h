/*
    This file is part of Sciscipy.

    Sciscipy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Foobar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
    
    Copyright (c) 2009, Vincent Guffens.
*/

#ifndef SCICONV_WRITE
#define SCICONV_WRITE

#include <Python.h>

#if NUMPY == 1
#include "util.h"
#include "numpy/arrayobject.h"
#endif

typedef enum {	NUMPY_ARRAY,
				LISTOFLIST, 
				LISTOFDOUBLE, 
				LISTOFSTRING, 
				TLIST
			   } WRITETYPE_t ;


struct sciconv_write_struct 
{
	int (*conv_func)(char*, PyObject*) ;	// Create a new variable in scilab  	
	int (*test_func)(PyObject*) ;			// Return one if this structure can handle the PyObject
	WRITETYPE_t write_type ;				// Identifier for the type
	struct sciconv_write_struct *next ;
} ;

// List of converter functions 
extern struct sciconv_write_struct* sciconv_write_list ;

void sciconv_write_init(void) ;



#endif 
