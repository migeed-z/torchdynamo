import torch
import torchdynamo
import torchdynamo.testing
from transformers import *
from timeit import default_timer as timer
from datetime import timedelta
from torchdynamo.helper_functions import  run_function, run_function_xglm
from torchdynamo.user_constraints import *

# 18 / 44
print('marian MT \n \n')
start = timer()
run_function(MarianMTModel, user_constraints_marian_mt)
end = timer()
print(timedelta(seconds=end-start))
