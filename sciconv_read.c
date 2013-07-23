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

/*
  Types defined in scilab:

  1  : real or complex constant matrix.
  2  : polynomial matrix.
  4  : boolean matrix.
  5  : sparse matrix.
  6  : sparse boolean matrix.
  8  : matrix of integers stored on 1 2 or 4 bytes
  9  : matrix of graphic handles
  10 : matrix of character strings.
  11 : un-compiled function.
  13 : compiled function.
  14 : function library.
  15 : list.
  16 : typed list (tlist)
  17 : mlist
  128 : pointer


 NOTE: the way numpy vectors and matrices are created leads
        to a memory leak.
*/

#include "sciconv_read.h"
#include "deallocator.h"
#include "util.h"

struct sciconv_read_struct *sciconv_read_list = NULL ;

PyObject *
sciconv_read (int *addr, int var_type)
{
    char er_msg[BUFSIZE] ;

    struct sciconv_read_struct *conv = sciconv_read_list ;

    while (conv)
    {
        if (conv->scitype == var_type)
        {
            return conv->conv_func(addr) ;
        }
        conv = conv->next ;
    }

    snprintf(er_msg, BUFSIZE, "Type %i not supported", var_type) ;
    PyErr_SetString(PyExc_TypeError, er_msg) ;
    return NULL ;


} ;


#if NUMPY == 1
static PyObject * create_numpyarray(double *cxtmp, int m, int n)
{
    PyObject *array ;
    npy_intp dim[2], mn ;

    if (m == 1 || n == 1)
    {
        mn = m * n ;

        array = PyArray_NewFromDescr(&PyArray_Type, \
                                     PyArray_DescrFromType(PyArray_DOUBLE), \
                                     1, \
                                     &mn, \
                                     NULL, \
                                     (void *) cxtmp, \
                                     NPY_OWNDATA | NPY_FARRAY, \
                                     NULL
                                    ) ;

        attach_deallocator(array, cxtmp) ;

        return array ;
    }

    dim[0] = m ;
    dim[1] = n ;

    array = PyArray_NewFromDescr(&PyArray_Type, \
                                 PyArray_DescrFromType(PyArray_DOUBLE), \
                                 2, \
                                 dim, \
                                 NULL, \
                                 (void *) cxtmp, \
                                 NPY_OWNDATA | NPY_FARRAY, \
                                 NULL
                                ) ;

    attach_deallocator(array, cxtmp) ;

    return array ;

}

static PyObject * create_cnumpyarray(double *cxtmp, double *cxtmp_img, int m, int n)
{
    PyObject * array ;
    int i, j  ;
    npy_intp dim[2], mn ;
    complex * cxtmp_transpose ;

    cxtmp_transpose = (complex*) malloc(2 * m * n * sizeof(complex));

    dim[0] = m ;
    dim[1] = n ;

    if (!cxtmp_transpose)
    {
        PyErr_SetString(PyExc_MemoryError, "out of memory") ;
        return NULL ;
    }

    for (i = 0; i < m; ++i)
        for (j = 0; j < n; ++j)
        {
            cxtmp_transpose[i * n + j][0] = cxtmp[j * m + i] ;
            cxtmp_transpose[i * n + j][1] = cxtmp_img[j * m + i] ;

        }

    if (m == 1 || n == 1)
    {
        mn = m * n ;


        array = PyArray_NewFromDescr(&PyArray_Type, \
                                     PyArray_DescrFromType(PyArray_CDOUBLE), \
                                     1, \
                                     &mn, \
                                     NULL, \
                                     (void *) cxtmp_transpose, \
                                     NPY_OWNDATA | NPY_CARRAY, \
                                     NULL
                                    ) ;
    }
    else

        array = PyArray_NewFromDescr(&PyArray_Type, \
                                     PyArray_DescrFromType(PyArray_CDOUBLE), \
                                     2, \
                                     dim, \
                                     NULL, \
                                     (void *) cxtmp_transpose, \
                                     NPY_OWNDATA | NPY_CARRAY, \
                                     NULL
                                    ) ;

    free(cxtmp) ;
    attach_deallocator(array, cxtmp_transpose) ;
    return array ;

}
#else

static PyObject * create_listmatrix(double *cxtmp,  double *cxtmp_img, int m, int n, int is_complex)
{
    int i, j ;
    PyObject *new_list, *new_line ;
    Py_complex new_complex ;


    if (m == 1 || n == 1)
    {
        new_list = PyList_New(m * n) ;

        for (i = 0 ; i < m * n ; i++)
        {
            if (cxtmp_img != NULL)
            {
                new_complex.real = cxtmp[i] ;
                new_complex.imag = cxtmp_img[i] ;
                PyList_SET_ITEM(new_list, i, Py_BuildValue("D", &new_complex)) ;
            }
            else
            {
                PyList_SET_ITEM(new_list, i, Py_BuildValue("d", cxtmp[i])) ;
            }
        }
    }
    else
    {
        new_list = PyList_New(m) ;
        for (i = 0 ; i < m ; i++)
        {
            new_line = PyList_New(n) ;
            for (j = 0 ; j < n ; j++)
                if (cxtmp_img != NULL)
                {
                    new_complex.real = cxtmp[j * m + i] ;
                    new_complex.imag = cxtmp_img[j * m + i] ;
                    PyList_SET_ITEM(new_line, j, Py_BuildValue("D", &new_complex)) ;
                }
                else
                {
                    PyList_SET_ITEM(new_line, j, Py_BuildValue("d", cxtmp[j * m + i])) ;
                }

            PyList_SET_ITEM(new_list, i, new_line) ;
        }
    }

    free(cxtmp) ;

    return new_list ;
};

#endif

/**
 * Type 1 : real or complex constant matrix.
 * @param name: the name of the scilab variable we want to read
 * @return: A list of list
*/
static PyObject * read_matrix(int *addr)
{

    int m, n ;
    SciErr sciErr ;
    double *cxtmp = NULL ;
    double *cx = NULL, *cx_img = NULL;
    double *cxtmp_img = NULL ;
    PyObject * matrix ;

    if (!isVarComplex(pvApiCtx, addr))
    {
        sciErr = getMatrixOfDouble(pvApiCtx, addr, &m, &n, NULL) ;
    }
    else
    {
        sciErr = getComplexMatrixOfDouble(pvApiCtx, addr, &m, &n, NULL, NULL) ;
    }

    if (sciErr.iErr)
    {
        PyErr_SetString(PyExc_TypeError, getErrorMessage(sciErr)) ;
        return 0;
    }


    cx = (double*)malloc((m * n) * sizeof(double));

    if (!cx)
    {
        PyErr_SetString(PyExc_MemoryError, "out of memory") ;
        return NULL ;
    }

    if (!isVarComplex(pvApiCtx, addr))
    {
        sciErr = getMatrixOfDouble(pvApiCtx, addr, &m, &n, &cxtmp) ;
        if (sciErr.iErr)
        {
            free(cx);
            PyErr_SetString(PyExc_TypeError, "Error in readmatrix") ;
            return 0;
        }

        memcpy(cx, cxtmp, sizeof(double) * n * m) ;
#if NUMPY == 1
        matrix = create_numpyarray(cx, m, n) ;
#else
        matrix = create_listmatrix(cx, NULL, m, n) ;
#endif

    }
    else
    {
        cx_img = (double*)malloc((m * n) * sizeof(double));

        if (!cx_img)
        {
            free(cx) ;
            free(cxtmp) ;
            PyErr_SetString(PyExc_MemoryError, "out of memory") ;
            return NULL ;
        }
        sciErr = getComplexMatrixOfDouble(pvApiCtx, addr, &m, &n, &cxtmp, &cxtmp_img) ;
        if (sciErr.iErr)
        {
            free(cx) ;
            free(cx_img);
            PyErr_SetString(PyExc_TypeError, "Error in readmatrix") ;
            return 0;
        }

        memcpy(cx, cxtmp, sizeof(double) * n * m) ;
        memcpy(cx_img, cxtmp_img, sizeof(double) * n * m) ;

#if NUMPY == 1
        matrix = create_cnumpyarray(cx, cx_img, m, n) ;
#else
        matrix = create_listmatrix(cx, cx_img, m, n) ;
#endif
        free(cx_img);
    }

    return matrix ;
}


/**
 * Type 10 : Matrix of string.
 * @param name: the name of the scilab variable we want to read
 * @return: A list of string
*/
static PyObject * read_string(int *addr)
{

    int m = 0, n = 0 ;
    int i = 0 ;
    int x = 0, y = 0 ;

    char ** variable_from_scilab = NULL ;
    SciErr sciErr;

    sciErr = getMatrixOfString(pvApiCtx, addr, &m, &n, NULL, NULL) ;
    if (sciErr.iErr)
    {
        PyErr_SetString(PyExc_TypeError, getErrorMessage(sciErr)) ;
        return 0;
    }

    int *piLen = (int*)malloc(sizeof(int) * m * n);

    PyObject *new_list ;
    sciErr = getMatrixOfString(pvApiCtx, addr, &m, &n, piLen, NULL) ;
    if (sciErr.iErr)
    {
        PyErr_SetString(PyExc_TypeError, getErrorMessage(sciErr)) ;
        return 0;
    }

    variable_from_scilab = (char **) malloc(sizeof(char*) * (m * n)) ;

    for (i = 0; i < m * n; i++)
    {
        variable_from_scilab[i] = (char*) malloc(sizeof(char) * (piLen[i])) ;
    }

    i = 0;
    new_list = PyList_New(m * n) ;
    sciErr = getMatrixOfString(pvApiCtx, addr, &m, &n, piLen, variable_from_scilab) ;
    if (sciErr.iErr)
    {
        PyErr_SetString(PyExc_TypeError, getErrorMessage(sciErr)) ;
        return 0;
    }

    for (x = 0; x < m; x++)
    {
        for (y = 0; y < n; y++)
        {
            char *tmpStr = variable_from_scilab[x * m + y] ;
            PyList_SET_ITEM(new_list, i, Py_BuildValue("s", tmpStr)) ;
            free(tmpStr) ;
            i++;
        }
    }

    return new_list ;
}

/**
 * Type 16 : tlist (typed list).
 *
 * A tlist x = tlist(['test','a','b'],12,'item')
 * is transformed in python in
 * x = { "__tlist_name" : "test",
 *       "a" : 12,
 *       "b" : "item",
 *     }
 *
 * @param tlist_address: the address of the scilab variable we want to read
 * @return: A dictionary
*/
static PyObject * read_tlist(int *tlist_address)
{
    SciErr sciErr ;
    PyObject *new_dict = NULL ;
    PyObject *key_list = NULL ;
    int nb_item = 0, i;

    sciErr = getListItemNumber(pvApiCtx, tlist_address, &nb_item) ;
    if (sciErr.iErr)
    {
        goto handle_error ;
    }

    new_dict = PyDict_New() ;
    for (i = 1 ; i <= nb_item; ++i)
    {
        PyObject *py_item ;
        int *item_address = NULL ;
        int sci_type = 0 ;

        sciErr = getListItemAddress(pvApiCtx, tlist_address, i, &item_address) ;
        if (sciErr.iErr)
        {
            goto handle_error ;
        }
        sciErr = getVarType(pvApiCtx, item_address, &sci_type) ;
        if (sciErr.iErr)
        {
            goto handle_error ;
        }
        py_item = sciconv_read (item_address, sci_type) ;
        if (i == 1)
        {
            if (sci_type != 10)
            {
                PyErr_SetString(PyExc_TypeError, "First tlist item must be string") ;
                return 0 ;
            }

            key_list = py_item ;
            PyDict_SetItem(new_dict, Py_BuildValue("s", TLIST_NAME), \
                           PyList_GetItem(key_list, i - 1)) ;

        }
        else
        {
            PyObject *next_item = NULL ;
            next_item = PyList_GetItem(key_list, i - 1) ;
            if (next_item == NULL)
            {
                PyErr_SetString(PyExc_TypeError, "Cannot read tlist (wrong number of key)") ;
                return 0 ;
            }
            PyDict_SetItem(new_dict, PyList_GetItem(key_list, i - 1), py_item) ;
        }
    }


    return new_dict;

handle_error:
    PyErr_SetString(PyExc_TypeError, getErrorMessage(sciErr)) ;
    return 0;
}

/**
 * Add a new converter to the list
 * @param new_type: A scilab type number
 * @param func: The converter function
*/
static void sciconv_read_add(int new_type, PyObject * (*func)(char*))
{
    struct sciconv_read_struct *new_conv = \
                                           (struct sciconv_read_struct*) malloc(sizeof(struct sciconv_read_struct)) ;

    new_conv->scitype = new_type ;
    new_conv->conv_func = func ;

    if (sciconv_read_list == NULL)
    {
        sciconv_read_list = new_conv ;
        new_conv->next = NULL ;
        return ;
    }

    new_conv->next = sciconv_read_list->next ;
    sciconv_read_list->next = new_conv ;
}

/**
 * Initialization
 * Add all the known converters to the list
*/
void sciconv_read_init(void)
{
    // Most used should come last
    sciconv_read_add(16, read_tlist) ;
    sciconv_read_add(10, read_string) ;
    sciconv_read_add(1, read_matrix) ;

}
