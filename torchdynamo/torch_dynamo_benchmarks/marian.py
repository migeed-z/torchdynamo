import torch
import torchdynamo
import torchdynamo.testing
from transformers import *
from timeit import default_timer as timer
from datetime import timedelta
from torchdynamo.helper_functions import  run_function
from torchdynamo.user_constraints import *


# 18 / 44
start = timer()
print('marian \n \n')
run_function(MarianModel, user_constraints_marian_mt)
end = timer()
print(timedelta(seconds=end-start))


