import torch
import torchdynamo
import torchdynamo.testing
from transformers import *
from timeit import default_timer as timer
from datetime import timedelta
from torchdynamo.helper_functions import  run_function, run_function_xglm
from torchdynamo.user_constraints import *

# 25 / 47
print('m2m100 \n \n')
start = timer()
run_function(M2M100Model, user_constraints_M2M100Model)
end = timer()
print(timedelta(seconds=end-start))








