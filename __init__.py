from krezi.common import *
fit_to_screen() # when reload is called on krezi, this will restore the cell length

from krezi.logging_util.file_logger import Logger
from krezi.mssql_util.pymssql_dao import mssql_dao
from krezi.posgresql_util.posgresql_util import pgdao
from krezi.multiprocessing_util.mp_class_util import ParallelExecutor
from krezi.stopwatch_util.stopwatch import Stopwatch



"""

if linux : write this in .bashrc
if mac : write this in .zshrc

cmd : PYTHONPATH="$PYTHONPATH:/Users/aadilzikre/Documents/Office PC/REBEL FOODS - Copy/Python/common functions az/"
cmd : echo PYTHONPATH

Module Funcs can be imported as 

1. import krezi --> runs __init__.py 
    common will be available as krezi.fit_to_screen() for ex
2. from krezi import * 
    common will be available as fit_to_screen() for ex
3. form krezi.common import *
4. import krezi.common as cf 
5. from krezi import Stopwatch

"""