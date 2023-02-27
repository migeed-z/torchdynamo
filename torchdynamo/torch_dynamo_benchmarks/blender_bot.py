import torch
import torchdynamo
import torchdynamo.testing
from transformers import *
from timeit import default_timer as timer
from datetime import timedelta
from torchdynamo.helper_functions import  run_function, run_function_xglm
from torchdynamo.user_constraints import *

# 16 / 35
start = timer()
print('blender bot \n \n')
run_function(BlenderbotSmallModel, user_constraints_blenderbot)
end = timer()
print(timedelta(seconds=end-start))







