# swarm_integration.py

from swarmlearning.client import SwarmClient, SLPlatforms, SLMergeMethod

class SwarmIntegration:
    def __init__(self, syncFrequency, minPeers, trainingContract, **kwargs):
        self.syncFrequency = syncFrequency
        self.minPeers = minPeers
        self.trainingContract = trainingContract
        self.platform = kwargs.get("platform", SLPlatforms.DEFAULT)
        self.mergeMethod = kwargs.get("mergeMethod", SLMergeMethod.AVERAGE)
        self.client = SwarmClient(syncFrequency=self.syncFrequency, minPeers=self.minPeers,
                                  trainingContract=self.trainingContract, platform=self.platform,
                                  mergeMethod=self.mergeMethod)

    def initialize_swarm(self):
        self.client.initialize()
        print("Swarm Learning initialized.")

    def sync_weights(self, model_state_dict):
        # Sync model weights across nodes
        merged_weights = self.client.syncWeights(model_state_dict)
        return merged_weights

    def finalize_swarm(self):
        self.client.finalize()
        print("Swarm Learning finalized.")