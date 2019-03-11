#!/usr/local/bin/python

from gpflow import settings
settings.session.intra_op_parallelism_threads = 1
settings.session.inter_op_parallelism_threads = 1

import pandas as pd
import numpy as np
import json
import os

import pandas as pd
import numpy as np
from GrandPrix import GrandPrix

import time as tm
checkpoints = {}

import dynclipy

#   ____________________________________________________________________________
#   Load data                                                               ####
task = dynclipy.main()
# task = dynclipy.main(
#   ["--dataset", "/code/example.h5", "--output", "/mnt/output"],
#   "/code/definition.yml"
# )


expression = task["expression"]
params = task["params"]
end_n = task["priors"]["end_id"]

if "timecourse_continuous" in task["priors"]:
  time = task["priors"]["timecourse_continuous"]
else:
  time = None

checkpoints["method_afterpreproc"] = tm.time()

#   ____________________________________________________________________________
#   Infer trajectory                                                        ####
# fit grandprix model, based on https://github.com/ManchesterBioinference/GrandPrix/blob/master/notebooks/Guo.ipynb
if end_n == 0:
  end_n = 1

if time is not None:
  pt_np, var_np = GrandPrix.fit_model(
    data = expression.values,
    n_latent_dims = end_n,
    n_inducing_points = params["n_inducing_points"],
    latent_var = params["latent_var"],
    latent_prior_mean = time.time,
    latent_prior_var = params["latent_prior_var"]
  )
else:
  pt_np, var_np = GrandPrix.fit_model(
    data = expression.values,
    n_latent_dims = end_n,
    n_inducing_points = params["n_inducing_points"],
    latent_var = params["latent_var"]
  )

checkpoints["method_aftermethod"] = tm.time()

#   ____________________________________________________________________________
#   Process output                                                          ####
# process to end_state_probabilities output format^
cell_ids = expression.index

pseudotime = pd.DataFrame({
  "cell_id": expression.index,
  "pseudotime": pt_np[:, 0]
})

end_state_probabilities = pd.DataFrame({"cell_id":expression.index})
for i in range(pt_np.shape[1] - 1):
  split_id = "split_" + str(i)

  probabilities = pt_np[:, 1]
  probabilities = (probabilities - min(probabilities))/(max(probabilities) - min(probabilities))

  end_state_probabilities[split_id + "_1"] = probabilities
  end_state_probabilities[split_id + "_2"] = 1-probabilities

# save
dataset = dynclipy.wrap_data(cell_ids = expression.index)
dataset.add_end_state_probabilities(
  end_state_probabilities = end_state_probabilities,
  pseudotime = pseudotime
)
dataset.add_timings(timings = timings)
dataset.write_output(task["output"])
