#ifndef _UTIL_H
#define _UTIL_H

#undef HAVE_STRERROR
#undef SIZEOF_LONG

#include "call_scilab.h"
#include "api_scilab.h"

#define BUFSIZE 1024

#define TLIST_NAME	"__tlist_name"

/*
 * This has to be defined when a numpy extension
 * is split accross multiple files
 */
#define PY_ARRAY_UNIQUE_SYMBOL UNIQUE

typedef double complex[2] ;

int read_sci_type(char *name) ;
int is_real(char *name) ;

void sci_debug(const char *format, ...) ;
void sci_error(const char *format, ...) ;

PyObject* create_list(PyObject *obj) ;
char *get_SCI(char*) ;
extern const int sci_max_len ;

#endif
