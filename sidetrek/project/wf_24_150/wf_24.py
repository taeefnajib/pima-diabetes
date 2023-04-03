import main
import os
import typing
from flytekit import workflow
from main import Hyperparameters
from main import run_wf

_wf_outputs=typing.NamedTuple("WfOutputs",run_wf_0=main.ANN_Model)
@workflow
def wf_24(_wf_args:Hyperparameters)->_wf_outputs:
	run_wf_o0_=run_wf(hp=_wf_args)
	return _wf_outputs(run_wf_o0_)