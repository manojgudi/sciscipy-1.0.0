// This modue provide three function to interact with a python interpretor:
//
//  - py_write : write a scilab variable in python
//  - py_read : read a python variable in scilab
//  - py_eval : evaluate a python string
//
// A convenience function to call into python is also provided

global GLOB_PY_INIT ;
GLOB_PY_INIT = %F ;

function py_write(pyname, scivar)
// TODO
endfunction

function sci_var = py_read(py_name)
  var_type = py_type(py_name)
  // Call the right function accoring to var_type
  sci_var = call('py_read_int_c', py_name, 1, 'c','out', [1,1], 2, 'i')
endfunction

function ptype = py_type(py_name)
// @param py_name: the name of a python variable
// @return: an integer, 
  ptype = 1
endfunction

function py_eval(eval_str)
  call('py_eval_c',eval_str, 1, 'c','out')
endfunction

function py_initialize()
	global GLOB_PY_INIT
	if GLOB_PY_INIT == %F
		call('py_initialize_c', 'out')
		GLOB_PY_INIT = %T ;
	end
endfunction

function py_finalize()
	call('py_finalize_c', 'out')
endfunction

function call_python(funcname, varargin)
  for i=1:length(varargin)
    arg = varargin(i)
    pywrite(printf("_tmp%i_", arg))
  end
endfunction

intf = [ 'py_initialize_c', 'py_finalize_c','py_eval_c', 'py_read_int_c'] ;

libn = ilib_for_link(intf, 'callpython.c','', 'c' ,'make' , ...
					'loader.sce' ,'callpython.lib','-LIBPATH:C:\Python26\libs', ...
					'-IC:\Python26\include') ;
					
exec('loader.sce') ;
py_initialize()

