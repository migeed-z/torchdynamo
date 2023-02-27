# This file is for TorchDynamo user constraints.
from torch.fx.experimental.migrate_gradual_types.z3_types import tensor_type, D
from z3 import z3


# Variable definitions are used for all models
s1, s2, s3, s4, s5, s6 = z3.Ints('x1 x2 x3 x4 x5 x6')
input = z3.Const(1, tensor_type)
input_embeds = z3.Const(3, tensor_type)
self_weights = z3.Const(3, tensor_type)
stack_0 = z3.Const(1, tensor_type)
attention_mask = z3.Const(2, tensor_type)
input_embeds_2 = z3.Const(2, tensor_type)
dimension_var2 = z3.Int(2)

#Constraints which will be propagated through program fragments
heuristic = [z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                     s1 > 0,
                     s2 > 1,
                     s2 < 2000])] * 20

heuristic2 = [z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                      s1 > 0,
                      s2 > 1,
                      s2 < 2000])] * 10


# false constraints to skip branches
false_constraints = [False] * 20


# The constraints we use for our models
user_constraints_M2M100Model = [z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2)), s1 > 0,  s2 > 1, s2 < 1024])]+ \
                               [z3.And([input_embeds_2 == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                        s1 > 0,
                                        s2 > 1,
                                        s2 < 2000,
                                        input_embeds_2 == stack_0])] * 6 + [False] + [z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                                                                              s1 > 0,
                                                                                              s2 > 1,
                                                                                              s2 < 2000])] * 7 + [False] + [z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                                                                                                                    s1 > 0,
                                                                                                                                    s2 > 1,
                                                                                                                                    s2 < 2000])] * 6 + [False]  + [z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                                                                                                                                                           s1 > 0,
                                                                                                                                                                           s2 > 1,
                                                                                                                                                                           s2 < 2000])] * 10 + [False] * 50
user_constraints_blenderbot = [False, False] \
                              + \
                              [z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, s3)),
                                       s1 > 0,
                                       s2 > 1,
                                       s2 < 2000])] * 8 + [False] + [z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, s3)),
                                                                             s1 > 0,
                                                                             s2 > 1,
                                                                             s2 < 2000])] * 13 + [False] * 40

user_constraints_XGLM = [z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2)), s1 > 0,  s2 > 1, s2 < 2000]),

                         True,

                         z3.And([input == tensor_type.tensor2(D(1, s1), D(1, s2)), s1 > 0,  s2 > 1, s2 < 2000,
                                 self_weights == tensor_type.tensor2(D(1, 2050), D(1, 1024))]),


                         z3.And([input_embeds == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                 s1 > 0,
                                 s2 > 1,
                                 s2 < 2000,
                                 input_embeds == stack_0]),

                         z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                 s1 > 0,
                                 s2 > 1,
                                 s2 < 2000,
                                 input_embeds == stack_0, input_embeds == attention_mask]),

                         z3.And([input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                 s1 > 0,
                                 s2 > 1,
                                 s2 < 2000]),
                         True,
                         True,
                         True,
                         True,
                         True,
                         True,
                         True]

user_constraints_marian_mt = [z3.And([input_embeds_2 == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                      s1 > 0,
                                      s2 > 1,
                                      s2 < 2000,
                                      input_embeds_2 == stack_0])] * 6 + [False] * 2 + [z3.And([input == tensor_type.tensor3(D(1,s1), D(1, s2), D(1, s3)),
                                                                                                input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                                                                                s1 > 0,
                                                                                                s2 > 1,
                                                                                                s2 < 2000])] * 6 + [False]  + \
                             [z3.And([input == tensor_type.tensor3(D(1,s1), D(1, s2), D(1, s3)),
                                      input == tensor_type.tensor3(D(1, s1), D(1, s2), D(1, 1024)),
                                      s1 > 0,
                                      s2 > 1,
                                      s2 < 2000])] * 6 + [False] * 40
