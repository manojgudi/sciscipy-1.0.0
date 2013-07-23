#include <Python.h>

void py_initialize_c()
{
	Py_Initialize() ;
}

void py_finalize_c()
{
	Py_Finalize() ;
}

void py_eval_c(char * exec_str)
{
	PyRun_SimpleString(exec_str) ;
}

// Read an integer value
// @in py_var: the name of the python var to read
// @out sci_var: the return variable
void py_read_int_c(char * py_var, int * sci_var)
{
	PyObject* main_module = PyImport_AddModule("__main__") ;
	PyObject* main_dict = PyModule_GetDict(main_module) ;

	if (PyDict_Contains(main_dict, PyString_FromString(py_var)))
	{
		PyObject * py_int = PyDict_GetItem(main_dict, PyString_FromString(py_var)) ;
		if (PyInt_Check(py_int)){
		
			*sci_var = PyInt_AS_LONG(py_int) ;
			return ;
		}
		// Handle not an int error
		return ;
	}
	// Handle var not found error
	*sci_var = 0 ;
}