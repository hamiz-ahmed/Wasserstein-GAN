import hpbandster
from hpbandster.distributed.worker import Worker
import hpbandster.distributed.utils
import ConfigSpace as CS
from config_space import get_config_space
import pickle
import os

import logging
logging.basicConfig(level=logging.INFO)


# starts a local nameserver
nameserver, ns_port = hpbandster.distributed.utils.start_local_nameserver()


# import the definition of the worker (could be in here as well, but is imported to reduce code duplication)
from worker import MyWorker


# Every run has to have a unique (at runtime) id.
# This needs to be unique for concurent runs, i.e. when multiple
# instances run at the same time, they have to have different ids
# Here we pick '0'
run_id = '0'


# starting the worker in a separate thread
w = MyWorker(nameserver=nameserver, run_id=run_id, ns_port=ns_port)
w.run(background=True)


# simple config space here: just one float between 0 and 1
config_space = get_config_space()

CG = hpbandster.config_generators.RandomSampling(config_space)

# min_budget=42500,
#                                      max_budget=500000,

# instantiating Hyperband with some minimal configuration
HB = hpbandster.HB_master.HpBandSter(config_generator = CG,
                                     run_id = run_id,
                                     eta=2,
                                     min_budget=42500,
                                     max_budget=500000,
                                     nameserver=nameserver,
                                     ns_port = ns_port,
                                     job_queue_sizes=(0,1),
                                     )

#min_budget=62500,
                                     #max_budget=1000000,

# runs one iteration if at least one worker is available
res = HB.run(1, min_n_workers=1)

# pickle result object
if not os.path.exists("logs/res"):
    os.makedirs("logs/res")
with open("logs/res/res_file", "wb+") as res_file:
    pickle.dump(res, res_file)

# shutdown the worker and the dispatcher
HB.shutdown(shutdown_workers=True)

print(res.get_incumbent_trajectory())

