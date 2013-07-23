/** deallocator.c
*
* Pattern described in http://blog.enthought.com/?p=62
*
*/

#include "deallocator.h"

void attach_deallocator(PyObject *array, void * mem)
{
    PyObject *newobj ;

    newobj =  _PyObject_New(&_MyDeallocType) ;
    ((struct _MyDeallocStruct *)newobj)->memory = mem ;
    PyArray_BASE(array) = newobj ;
    if (DEBUG_MEM_ALLOC == 1)
    {
        printf("ALLOCATED %p\n", mem) ;
    }
} ;

static void _mydealloc_dealloc(PyObject *self)
{
    if (DEBUG_MEM_ALLOC == 1)
    {
        printf("FREEING %p\n", ((struct _MyDeallocStruct*) self)->memory) ;
    }
    free(((struct _MyDeallocStruct*) self)->memory);
    self->ob_type->tp_free((PyObject *) self);
} ;

PyTypeObject _MyDeallocType =
{
    PyObject_HEAD_INIT(NULL)
    0,                              /*ob_size*/
    "mydeallocator",                /*tp_name*/
    sizeof(_MyDeallocObject),   	/*tp_basicsize*/
    0,                              /*tp_itemsize*/
    _mydealloc_dealloc,             /*tp_dealloc*/
    0,                         	/*tp_print*/
    0,                         	/*tp_getattr*/
    0,                         	/*tp_setattr*/
    0,                         	/*tp_compare*/
    0,                         	/*tp_repr*/
    0,                         	/*tp_as_number*/
    0,                         	/*tp_as_sequence*/
    0,                         	/*tp_as_mapping*/
    0,                         	/*tp_hash */
    0,                         	/*tp_call*/
    0,                         	/*tp_str*/
    0,                         	/*tp_getattro*/
    0,                         	/*tp_setattro*/
    0,                         	/*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,        	/*tp_flags*/
    "Internal deallocator object",  /* tp_doc */
} ;

