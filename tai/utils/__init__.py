from clu import metrics
import flax
import jax
from flax.training.train_state import TrainState
from flax import struct

@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    accuracy: metrics.Accuracy

class TrainState(TrainState):
    metrics: Metrics