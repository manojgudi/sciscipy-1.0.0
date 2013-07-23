"""
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


scilab module
=============

Provide an easy to use interface toward scilab

Usage:
>>> from scilab import scilab as sci
>>> x = [1,2,3]
>>> sci.disp(x)

    1.
    2.
    3.

With the help of the sciscipy module, you can also do:

>>> sciscipy.eval("function y = func(x) ; y = x*x ; endfunction")
>>> sci.func(2)
array([ 4.])


Internals
=========

It uses macrovar ou fun2string to discover the number of
output args


@author: vincent.guffens@gmail.com
"""

from sciscipy import write, read, eval
from threading import Thread
from ConfigParser import ConfigParser

import os
import sys
import time

DFLT_CONFIG = "scilab.cfg"
SECTION_CONFIG = "KNOWN FUNC"

# Type 130 functions do not
# work with macrovar so their
# output vars is hardcoded here
__known_func = {}


class ScilabError(Exception):
    """ Define an exception class """
    pass

def update_scilab_func(filename = None):
    """
    Look for filename and update the dictionary L{__known_func}
    filename is a python config file
    """
    assert isinstance(filename, (type(None), str)), "Wrong filename"

    # Search first in the current path
    if filename == None and os.path.isfile(DFLT_CONFIG):
        filename = os.path.join (DFLT_CONFIG)

    # Search here too: share/sciscipy/scilab.cfg
    if filename == None:
        filename = os.path.join (os.path.dirname(__file__), "..", "..", "..", "share", "sciscipy", DFLT_CONFIG)

    if filename == None:
        filename = os.path.join (sys.prefix, 'share', 'sciscipy', DFLT_CONFIG)


    if not os.path.exists(filename):
        raise ValueError, "can not open file: " + filename

    parser = ConfigParser()
    parser.read(filename)

    if not parser.has_section(SECTION_CONFIG):
        raise ValueError, "Invalid config file"

    items = parser.items(SECTION_CONFIG)

    for new_func, value in items:
            __known_func[new_func] = int(value)


def run_scilab_cmd(cmd_str):
    """ Defines the Scilab start command (with error handle) """
    new_cmd = "_ier_ = execstr('%s', 'errcatch'); _er_msg_ = lasterror() ;" % cmd_str
    eval(new_cmd)
    ier = read("_ier_")
    if ier != 0 and ier != [0]:
        lasterror = read("_er_msg_")
        raise ScilabError, lasterror


def find_scilab_type(var_name):
    """
    Find the scilab type of var_name

    @param var_name: name of a scilab variable
    @type var_name: string
    @return: type(var_name)
    """
    if type(var_name) != type(""):
        raise TypeError, "var_name must be a string"

    run_scilab_cmd("_tmp1_ = type(" + var_name + ")")
    res = read("_tmp1_")
    eval("clear _tmp1_")

    return res[0]

def find_output_param(macro_name):
    """
    Find out the number of output param of macro_name


    First we look in the __known_func dico to see
    if we have a special case for that macro. If not,
    we use macrovar for type 13 functions. Otherwise,
    we return 1.

    @param macro_name: the name of a scilab macro
    @type macro_name: string
    @return: number of ouput param of macro_name
    @rtype: integer
    """
    if type(macro_name) != type(""):
        raise TypeError, "macro_name must be a string"

    if macro_name in __known_func.keys():
        return __known_func[macro_name]

    if find_scilab_type(macro_name) == 13:
        eval("_tmp1_ = macrovar(" + macro_name + ");")
        eval("_tmp2_ = length(length(_tmp1_(2)))")
        res = read("_tmp2_")
        eval("clear _tmp1_, _tmp2_")
        return int(res[0])

    return 1


class Functor(object):
    """
    The attribute 'name' is the name
    of the function to call in scilab
    """

    def __init__(self, name):
        if type(name) != type(""):
            raise TypeError, "name must be a string"

        self.name = name

    def __call__(self, *args):
        """
        TODO: add a named argument outp=...
            if you want to force the number of output arguments
        """
        cmd = self.name + "("

        in_args = []
        for (i, arg) in enumerate(args):
            arg_name = "__arg" + str(i)
            in_args += [arg_name]
            write(arg_name, arg)

        out = find_output_param(self.name)

        out_args = []
        for i in range(out):
            out_args += ["__out" + str(i)]

        if out != 0:
            cmd = "[%s] = %s(%s)" % (",".join(out_args),
                                   self.name,
                                   ",".join(in_args))
        else:
            cmd = "%s(%s)" % (self.name, ",".join(in_args))


        run_scilab_cmd(cmd)

        if out == 0:
            return None

        res = []
        for i in range(out):
            item = read("__out" + str(i))
            res += [item]
        
        if len(res) == 1:
            return res[0]
        else:
            return tuple(res)


class Scilab(object):
    """
    This class can call any scilab function (yeah!)

    Just instanciate an object of this class and call any
    method to call equivalent scilab function.

    >>> sci = Scilab()
    >>> from scilab import Scilab
    >>> sci = Scilab()
    >>> sci.zeros(2,2)
    [[0.0, 0.0], [0.0, 0.0]]
    >>>
    """

    def __getattr__(self, name):
        return Functor(name)

class ScilabThread(Thread):
        """ Defines the Scilab thread to start """
        def __init__(self, func):
                Thread.__init__(self)
                self.func = func
                self.daemon = True

        def run(self):
                self.func()

def scipoll():
    HOW_LONG = 0.1 # sec
    while 1:
        eval("")
        time.sleep(HOW_LONG)


# Update the dictionary
update_scilab_func()

# Create a convenience Scilab object
scilab = Scilab()

# Run the polling thread
poll_thread = ScilabThread(scipoll)
poll_thread.start()
