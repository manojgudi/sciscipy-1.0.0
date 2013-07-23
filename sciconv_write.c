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

#include "sciconv_write.h"
#include "util.h"
#include "Scierror.h"

struct sciconv_write_struct *sciconv_write_list = NULL ;

/** list of string
 *
 * WARNING: This will not work with Python 3.0
 */
static int write_listofstring(char *name, PyObject *obj)
{
    int i, n ;
    int tot_size = 0;
    PyObject *item ;
    char *str_item, *buff, *ptr ;

    if (!PyList_Check(obj))
    {
        obj = create_list(obj) ;
    }

    n = PyList_Size(obj) ;

    for (i = 0; i < n; ++i)
    {
        item = PyList_GetItem(obj, i) ;
        str_item = PyString_AsString(item) ;
        tot_size += strlen(str_item) ;
    }

    buff = (char *) malloc((strlen(name) + tot_size + 3 * n + 3) * sizeof(char)) ;
    ptr = buff ;
    strcpy(ptr, name) ;
    ptr += strlen(name) ;

    strcpy(ptr, "=[") ;
    ptr += 2 ;

    for (i = 0; i < n; ++i)
    {
        item = PyList_GetItem(obj, i) ;
        str_item = PyString_AsString(item) ;
        strcpy(ptr, "'") ;
        ptr++ ;
        strcpy(ptr, str_item) ;
        ptr += strlen(str_item) ;
        strcpy(ptr, "'") ;
        ptr++ ;
        if (i != n - 1)
        {
            strcpy(ptr, ",") ;
            ptr++ ;
        }
    }

    strcpy(ptr, "]") ;
    SendScilabJob(buff);

    return 1 ;
}

/**
 */
static int test_listofstring(PyObject *obj)
{
    PyObject *item ;
    int n ;

    if (PyString_Check(obj))
    {
        sci_debug("[sciconv_write] Match for list of string\n") ;
        return 1 ;
    }

    if (PyList_Check(obj))
    {
        n = PyList_Size(obj) ;
        if (n == 0)
        {
            return -1 ;
        }

        item = PyList_GetItem(obj, 0) ;

        if (PyString_Check(item))
        {
            sci_debug("[sciconv_write] Match for list of string\n") ;
            return 1 ;
        }
    }

    return -1 ;
}

/**
 * @param name: the name of the scilab variable we want to create
 * @return: negative on failure
*/
static int write_listoflist(char *name, PyObject *obj)
{
    int i, j, m, n ;
    double *new_vec ;
    double *new_vec_img ;
    PyObject *item, *element ;
    int is_complex_found = 0 ;
    SciErr sciErr;

    m = PyList_Size(obj) ;
    item = PyList_GetItem(obj, 0) ;
    n = PyList_Size(item) ;

    new_vec = (double*) calloc(sizeof(double) * m * n, 1) ;
    if (!new_vec)
    {
        return -1 ;
    }

    new_vec_img = (double*) calloc(sizeof(double) * m * n, 1) ;
    if (!new_vec_img)
    {
        free(new_vec);
        return -1 ;
    }

    for (i = 0; i < m; i++)
    {
        item = PyList_GetItem(obj, i) ;

        for (j = 0; j < n; j++)
        {
            element = PyList_GetItem(item, j) ;

            if (PyComplex_Check(element))
            {
                is_complex_found = 1 ;
                new_vec[j * m + i] = PyComplex_RealAsDouble(element) ;
                new_vec_img[j * m + i] = PyComplex_ImagAsDouble(element) ;
                continue ;
            }

            if (PyFloat_Check(element) || PyLong_Check(element) || PyInt_Check(element))
            {
                new_vec[j * m + i] = PyFloat_AsDouble(element) ;
                continue ;
            }

            sci_debug("[write_listoflist] something found" \
                      "that is not real or complex") ;
            free(new_vec) ;
            free(new_vec_img) ;
            return -1 ;
        }
    }

    if (is_complex_found)
    {
        sciErr = createNamedComplexMatrixOfDouble(pvApiCtx, name, m, n, new_vec, new_vec_img);
        free(new_vec_img);
        free(new_vec);
        if (sciErr.iErr)
        {
            printError(&sciErr, 0);
            Scierror(999, "Cannot create complex variable '%s'.\n", name);
            return 0;
        }

    }
    else
    {
        sciErr = createNamedMatrixOfDouble(pvApiCtx, name, m, n, new_vec);
        free(new_vec) ;
        free(new_vec_img) ;
        if (sciErr.iErr)
        {
            PyErr_SetString(PyExc_TypeError, "Error in Writematrix");
            return 0;
        }
    }
    return 1 ;
}

static int test_listoflist(PyObject *obj)
{
    int n ;
    PyObject *item, *el ;


    if (!PyList_Check(obj))
    {
        return -1 ;
    }
    n = PyList_Size(obj) ;
    if (n == 0)
    {
        return -1 ;
    }

    item = PyList_GetItem(obj, 0) ;

    if (!PyList_Check(item) || PyList_Size(item) == 0)
    {
        return -1 ;
    }

    el = PyList_GetItem(item, 0) ;

    /* Only the first element is checked, the converter
        will fail later on if all items are not real or
        complex (This is for performance)
    */
    if (PyFloat_Check(el) || PyLong_Check(el) || PyComplex_Check(el) || PyInt_Check(el))
    {
        sci_debug("[sciconv_write] Match for list of list\n") ;
        return 1 ;
    }
    else
    {
        return -1 ;
    }
}

static int write_listofdouble(char *name, PyObject *obj)
{
    int i, m ;
    int n = 1 ;
    double *new_vec ;
    double *new_vec_img ;
    PyObject *element ;
    int is_complex_found = 0 ;
    SciErr sciErr;

    if (!PyList_Check(obj))
    {
        obj = create_list(obj) ;
    }

    m = PyList_Size(obj) ;

    new_vec = (double*) calloc(sizeof(double) * m, 1) ;

    if (!new_vec)
    {
        return -1 ;
    }

    new_vec_img = (double*) calloc(sizeof(double) * m, 1) ;

    if (!new_vec_img)
    {
        free(new_vec);
        return -1 ;
    }

    for (i = 0; i < m; i++)
    {
        element = PyList_GetItem(obj, i) ;

        if (PyComplex_Check(element))
        {
            is_complex_found = 1 ;
            new_vec[i] = PyComplex_RealAsDouble(element) ;
            new_vec_img[i] = PyComplex_ImagAsDouble(element) ;
            continue ;
        }

        if (PyFloat_Check(element) || PyLong_Check(element) || PyInt_Check(element))
        {
            new_vec[i] = PyFloat_AsDouble(element) ;
            continue ;
        }

        sci_debug("[write_listofdouble] something found" \
                  "that is not real or complex") ;
        free(new_vec) ;
        free(new_vec_img) ;
        return -1 ;
    }


    if (is_complex_found)
    {
        sciErr = createNamedComplexMatrixOfDouble(pvApiCtx, name, n, m, new_vec, new_vec_img);
        free(new_vec);
        free(new_vec_img);
        if (sciErr.iErr)
        {
            printError(&sciErr, 0);
            Scierror(999, "Cannot create complex variable '%s'.\n", name);
            return 0;
        }
    }
    else
    {
        sciErr = createNamedMatrixOfDouble(pvApiCtx, name, n, m, new_vec);
        free(new_vec);
        free(new_vec_img);
        if (sciErr.iErr)
        {
            PyErr_SetString(PyExc_TypeError, "Error in Writematrix") ;
            return 0;
        }
    }

    return 1 ;

}

static int test_listofdouble(PyObject *obj)
{
    PyObject *item ;
    int n ;

    if (PyFloat_Check(obj) || PyLong_Check(obj) || PyComplex_Check(obj) || PyInt_Check(obj))
    {
        sci_debug("[sciconv_write] Match for list of double\n") ;
        return 1 ;
    }

    if (PyList_Check(obj))
    {
        n = PyList_Size(obj) ;
        if (n == 0)
        {
            return -1 ;
        }

        item = PyList_GetItem(obj, 0) ;

        if (PyFloat_Check(item) || PyLong_Check(item) || PyComplex_Check(item) || PyInt_Check(item))
        {
            sci_debug("[sciconv_write] Match for list of double\n") ;
            return 1 ;
        }
    }

    return -1 ;
}

#if NUMPY == 1
static int write_numpy(char *name, PyObject *obj)
{

    PyArrayObject * array = (PyArrayObject *) obj ;
    double * data ;
    double * data_img ;
    int i, j, m, n ;
    complex * item ;
    SciErr sciErr;

    // TODO: add support for 1D array

    if (array->nd != 1 && array->nd != 2)
    {
        sci_debug("[sciconv_write] Only 1D and 2D array are supported\n") ;
        return -1 ;
    }

    if (array->nd == 1)
    {
        m = array->dimensions[0] ;
        n = 1 ;
    }
    else
    {
        m = array->dimensions[0] ;
        n = array->dimensions[1] ;
    }

    if ((array->descr->type_num == PyArray_DOUBLE) || \
            (array->descr->type_num == PyArray_INT) )
    {

        data = (double*) malloc(m * n * sizeof(double)) ;

        if (!data)
        {
            sci_error("[sciconv_write] out of memory\n") ;
            return -1 ;
        }

        for (i = 0; i < m ; i++)
            for (j = 0; j < n ; j++)
                data[j * m + i] = *(double*)(array->data + i * array->strides[0] + \
                                             j * array->strides[1]) ;

        sciErr = createNamedMatrixOfDouble(pvApiCtx, name, m, n, data);
        free(data);
        if (sciErr.iErr)
        {
            PyErr_SetString(PyExc_TypeError, "Error in Writematrix") ;
            return 0;
        }

        return 1 ;
    }

    if (array->descr->type_num == PyArray_CDOUBLE)
    {
        data = (double*) malloc(m * n * sizeof(double)) ;
        data_img = (double*) malloc(m * n * sizeof(double)) ;

        if (!data)
        {
            sci_error("[sciconv_write] out of memory\n") ;
            free(data_img);
            return -1 ;
        }

        for (i = 0; i < m ; i++)
            for (j = 0; j < n ; j++)
            {

                item = (complex*)(array->data + i * array->strides[0] + \
                                  j * array->strides[1]) ;
                data[j * m + i] = (*item)[0] ;
                data_img[j * m + i] = (*item)[1] ;
            }

        sciErr = createNamedComplexMatrixOfDouble(pvApiCtx, name, m, n, data, data_img);
        if (sciErr.iErr)
        {
            printError(&sciErr, 0);
            free(data);
            free(data_img);
            Scierror(999, "Cannot create complex variable '%s'.\n", name);
            return 0;
        }


        free(data) ;
        free(data_img) ;
        return 1 ;
    }

    sci_debug("[sciconv_write] Array type not supported\n") ;
    return -1 ;
}

static int test_numpy(PyObject *obj)
{
    if (PyArray_Check(obj))
    {
        sci_debug("[sciconv_write] Match for numpy array\n") ;
        return 1 ;
    }
    else
    {
        return -1 ;
    }
}
#endif

static int write_tlist(char *name, PyObject *obj)
{
    int nb_item ;
    int *tlist_address ;
    SciErr sciErr ;
    nb_item = PyDict_Size(obj) - 1 ;
    sciErr = createNamedTList(pvApiCtx, name, nb_item, &tlist_address) ;
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    PyObject *py_str_to_create = PyString_FromString("[") ;
    PyObject *py_value_str = PyString_FromString("") ;
    printf("Entering write 1\n") ;
    if (sciErr.iErr)
    {
        PyErr_SetString(PyExc_TypeError, getErrorMessage(sciErr)) ;
        return -1;
    }
    printf("Entering write 2\n") ;

    //  py_str_tocreate =
    //    ['a_name', 'var1', ..., 'varN']
    while (PyDict_Next(obj, &pos, &key, &value))
    {
        char *str_key = NULL ;
        if (!PyString_Check(key))
        {
            return -1 ;
        }

        str_key = PyString_AsString(key) ;

        if (strcmp(str_key, TLIST_NAME) == 0)
        {
            if (!PyString_Check(value))
            {
                return -1 ;
            }
            PyString_Concat(&py_str_to_create,  PyString_FromString("\"")) ;
            PyString_Concat(&py_str_to_create, value) ;
            PyString_Concat(&py_str_to_create,  PyString_FromString("\"")) ;
        }
        printf("Entering write i\n") ;
    }
    printf("Entering write 3\n") ;
    pos = 0 ;
    while (PyDict_Next(obj, &pos, &key, &value))
    {
        char *str_key = NULL ;
        if (!PyString_Check(key))
        {
            return -1 ;
        }

        str_key = PyString_AsString(key) ;
        if (strcmp(str_key, TLIST_NAME) != 0)
        {
            char rnd_name[BUFSIZE] ;
            PyString_Concat(&py_str_to_create,  PyString_FromString(",\"")) ;
            PyString_Concat(&py_str_to_create, key) ;
            PyString_Concat(&py_str_to_create,  PyString_FromString("\"")) ;
            // TODO
            // write(rnd_name, value)
            snprintf(rnd_name, BUFSIZE - 1, ",rnd_var__%i", rand()) ;
            PyString_Concat(&py_value_str, PyString_FromString(rnd_name)) ;
        }
    }
    PyString_Concat(&py_str_to_create, PyString_FromString("]")) ;
    printf("%s = tlist(%s%s)\n", name, PyString_AsString(py_str_to_create), PyString_AsString(py_value_str)) ;
    // creates a string name = tlist(['a_name', 'var1', ..., 'varN'], var1,..., varN)
    // eval the string

    return 1 ;
}

static int test_dict_tlist(PyObject *obj)
{
    PyObject *py_list_name = PyString_FromString(TLIST_NAME) ;
    if (PyDict_Check(obj) && PyDict_Contains(obj, py_list_name))
    {
        sci_debug("[sciconv_write] Match for tlist\n") ;
        Py_DECREF(py_list_name) ;
        return 1 ;
    }
    else
    {
        Py_DECREF(py_list_name) ;
        return -1 ;
    }
}


/**
 * Add a new converter to the list
 * @param new_type: A scilab type number
 * @param func: The converter function
*/
static void sciconv_write_add(int (*test_func)(PyObject*), int(*func)(char*, PyObject *), WRITETYPE_t id)
{
    struct sciconv_write_struct *new_conv = \
                                            (struct sciconv_write_struct*) malloc(sizeof(struct sciconv_write_struct)) ;

    new_conv->test_func = test_func ;
    new_conv->conv_func = func ;
    new_conv->write_type = id ;
    if (sciconv_write_list == NULL)
    {
        sciconv_write_list = new_conv ;
        new_conv->next = NULL ;
        return ;
    }

    new_conv->next = sciconv_write_list->next ;
    sciconv_write_list->next = new_conv ;
}

/**
 * Initialization
 * Add all the known converter to the list
*/
void sciconv_write_init(void)
{
    // The one added first is the one tested first
    // so the order can be important
#if NUMPY == 1
    sciconv_write_add(test_numpy, write_numpy, NUMPY_ARRAY) ;
#endif
    sciconv_write_add(test_listoflist, write_listoflist, LISTOFLIST) ;
    sciconv_write_add(test_listofdouble, write_listofdouble, LISTOFDOUBLE) ;
    sciconv_write_add(test_listofstring, write_listofstring, LISTOFSTRING) ;
    sciconv_write_add(test_dict_tlist, write_tlist, TLIST) ;
}

