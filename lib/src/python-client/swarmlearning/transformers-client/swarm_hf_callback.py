# swarm_hf_callback.py

from transformers import TrainerCallback
from swarm_integration import SwarmIntegration

class SwarmHuggingFaceCallback(TrainerCallback):
    def __init__(self, syncFrequency, minPeers, trainingContract=None, **kwargs):
        super().__init__()
        self.swarm_integration = SwarmIntegration(syncFrequency, minPeers, trainingContract, **kwargs)

    def on_train_begin(self, args, state, control, **kwargs):
        # Initialize Swarm Learning at the start of training
        self.swarm_integration.initialize_swarm()

    def on_epoch_end(self, args, state, control, **kwargs):
        # Sync weights at the end of each epoch
        model = kwargs.get('model')
        if model is not None:
            model_state_dict = model.state_dict()
            merged_weights = self.swarm_integration.sync_weights(model_state_dict)
            model.load_state_dict(merged_weights)

    def on_train_end(self, args, state, control, **kwargs):
        # Finalize Swarm Learning after training
        self.swarm_integration.finalize_swarm()