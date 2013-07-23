#ifndef _DEALLOCATOR_H
#define _DEALLOCATOR_H

#include <Python.h>

#if NUMPY == 1
#include <numpy/arrayobject.h>


#define DEBUG_MEM_ALLOC	0

struct _MyDeallocStruct
{
	PyObject_HEAD
	void *memory ;
} ;


extern struct _MyDeallocStruct _MyDeallocObject ; 
extern PyTypeObject _MyDeallocType ;
extern void attach_deallocator(PyObject *, void *) ;

#endif
#endif