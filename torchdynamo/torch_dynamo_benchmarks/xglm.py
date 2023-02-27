import torch
import torchdynamo
import torchdynamo.testing
from timeit import default_timer as timer
from datetime import timedelta
from torchdynamo.helper_functions import run_function_xglm

# 5 / 5
print('XGLM \n \n')
start = timer()
run_function_xglm()
end = timer()
print(timedelta(seconds=end-start))