#######################################################################
## (C)Copyright 2023 Hewlett Packard Enterprise Development LP
## Licensed under the Apache License, Version 2.0 (the "License"); you may
## not use this file except in compliance with the License. You may obtain
## a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
## License for the specific language governing permissions and limitations
## under the License.
#######################################################################

##################################################################
# This file is the main entry point for Swarm Learning for Hugging Face Transformers.
# Users can integrate Swarm framework into their model code by creating an 
# instance of the SwarmCallback class and calling its methods 
# at different phases of training.
##################################################################

from transformers import TrainerCallback
from swarmlearning.client.swarm import SwarmCallbackBase, SLPlatforms

# Default Training contract used for learning if not specified by user. 
# Any update to default contract needs similar modifications 
# in all applicable ML platforms (TF, PYT, etc)
DEFAULT_TRAINING_CONTRACT = 'defaultbb.cqdb.sml.hpe'


# TODO - Why multiple inheritance ?   We don't do that for other platforms...
class SwarmCallback(SwarmCallbackBase, TrainerCallback):
    '''
    This is the customized callback class sub-classed from 
    SwarmCallbackBase class and TrainerCallback. It implements 
    different swarm functionalities for Hugging Face Transformers.
    '''

    class HuggingFaceContext:
        def __init__(self, model):
            self.model = model

    def __init__(self, syncFrequency, minPeers, trainingContract=DEFAULT_TRAINING_CONTRACT, **kwargs):
        '''
        Initializes the Swarm learning parameters and Hugging Face context.
        '''
        TrainerCallback.__init__(self)
        SwarmCallbackBase.__init__(self, syncFrequency, minPeers, trainingContract, kwargs)
        self._verifyAndSetPlatformContext(kwargs)
        self._swarmInitialize()
        
        # TODO: Check if its applicable and fix it.
        # lossFunction and lossFunctionArgs requested through swarm callback should match with 
        # loss function class defined in pyTorch
        # https://pytorch.org/docs/stable/nn.html#loss-functions
        self.lossFunction = kwargs.get('lossFunction', None)
        self.lossFunctionArgs = kwargs.get('lossFunctionArgs', None)
        
        # metricFunction and metricFunctionArgs requested through swarm callback should match with 
        # metric function class defined in torchmetrics.
        # https://torchmetrics.readthedocs.io/en/stable/all-metrics.html
        self.metricFunction = kwargs.get('metricFunction', None)
        self.metricFunctionArgs = kwargs.get('metricFunctionArgs', None)
        
        if(self.valData == None):
            self.logger.info("=============================================================")
            self.logger.info("WARNING: valData is not available to compute Loss and metrics")
            self.logger.info("=============================================================")

    def on_train_begin(self, args, state, control, **kwargs):
        '''
        Hugging Face specific on_train_begin implementation.
        '''
        self._swarmOnTrainBegin()
    
    def on_batch_end(self, batch=None):
        '''
        Hugging Face specific on_batch_end implementation
        '''
        self._swarmOnBatchEnd()
        
    def on_epoch_end(self, args, state, control, **kwargs):
        '''
        Hugging Face specific on_epoch_end implementation.
        Syncs model weights at the end of each epoch.
        '''
        model = kwargs.get('model')
        if model:
            model_state_dict = model.state_dict()
            merged_weights = self._syncWeights(model_state_dict)
            model.load_state_dict(merged_weights)

    def on_train_end(self, args, state, control, **kwargs):
        '''
        Hugging Face specific on_train_end implementation.
        Finalizes Swarm Learning after training.
        '''
        self._swarmOnTrainEnd()

    def _verifyAndSetPlatformContext(self, params):
        '''
        Hugging Face specific platform context initialization.
        '''
        ml_platform = params.get('ml_platform', SLPlatforms.HUGGINGFACE.name)
        if ml_platform not in [SLPlatforms.HUGGINGFACE.name]:
            self._logAndRaiseError("Invalid ML platform type: %s" % ml_platform)
        self.mlPlatform = SLPlatforms[ml_platform]
        self.model = params.get('model', None)
        if self.model is None:
            self._logAndRaiseError("Hugging Face model is None")
        else:
            self.__setMLContext(model=self.model)

    def _getValidationDataForAdaptiveSync(self, valData, valBatchSize):
        '''
        Hugging Face specific implementation of abstract method
        _getValidationDataForAdaptiveSync in SwarmCallbackBase class.
        '''
        valGen = validationSteps = valX = valY = valSampleWeight = None
        
        if(not valData):
            #valData is an optional parameter if not available, then performance data won't be supported
            return valGen, validationSteps, valX, valY, valSampleWeight
        
        # TODO - Need to check if it works.. 
        # No need to unpack valData for pyTorch. 
        # pyTorch supports only DataLoader object as valData
        # expecting valdata of type "torch.utils.data.DataLoader"
        if(not isinstance(valData, torch.utils.data.DataLoader)):
            self._logAndRaiseError("adsValData type passed is %s, not matching with torch.utils.data.DataLoader" %(type(valData)))
        
        # Following return of all Nones is to be in consistency with other ML platforms
        # valData object will be unpacked during loss/metrics computations
        return valGen, validationSteps, valX, valY, valSampleWeight


    def _saveModelWeightsToDict(self):
        '''
        Hugging Face specific implementation of saving model weights to a dictionary.
        '''
        inDict = {}
        self.weightNames = []
        model = self.mlCtx.model
        for wTensor in model.state_dict():
            inDict[wTensor] = model.state_dict()[wTensor].cpu().numpy()
            self.weightNames.append(wTensor)
        return inDict

    def _loadModelWeightsFromDict(self, inDict):
        '''
        Hugging Face specific implementation of loading model weights from a dictionary.
        '''
        model = self.mlCtx.model
        tempDict = {k: torch.Tensor(inDict[k]) for k in self.weightNames}
        model.load_state_dict(tempDict, strict=False)

    def _calculateLocalLossAndMetrics(self):
        '''
        Hugging Face specific implementation of calculating local loss and metrics.
        '''
        valLoss = 0
        totalMetrics = 0
        model = self.mlCtx.model

        if self.valData is None:
            return valLoss, totalMetrics

        # Logic for computing loss and metrics similar to the one used in pyt.py
        # Adjustments will be made based on Hugging Face's methods and structures.

        return valLoss, totalMetrics

    def __setMLContext(self, **params):
        ctx = SwarmCallback.HuggingFaceContext(params['model'])
        self.logger.debug("Initialized Hugging Face context for Swarm")
        self.mlCtx = ctx
