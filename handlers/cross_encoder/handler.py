import torch
import logging
import os
import sentence_transformers
from ts.torch_handler.base_handler import BaseHandler
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)
logger.info("sentence_transformers version %s", sentence_transformers.__version__)


class ModelHandler(BaseHandler):
    def initialize(self, context):
        """
        Initialize function loads the model

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.

        Raises:
            RuntimeError: Raises the Runtime error when the model is missing
        """

        properties = context.system_properties
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")

        # use GPU if available
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        logger.info(f'Using device {self.device}')

        # load the Cross_encoding model
        model_file = self.manifest['model']['modelFile']
        model_path = os.path.join(model_dir, model_file)

        if os.path.isfile(model_path):
            self.model = CrossEncoder(model_dir, device=self.device)
            logger.info(f'Successfully loaded cross_encoder model from {model_file}')
        else:
            raise RuntimeError('Missing the cross_encoder model file')

        self.initialized = True

    def preprocess(self, requests):
        """
        Extracts the question and references from the request

        Args:
            requests: A list containing a dictionary, might be in the form
            of [{'body': json_file}] or [{'data': json_file}]]
        Returns:
            the list of strings.
        """

        # unpack the data
        data = requests[0].get('body')
        if data is None:
            data = requests[0].get('data')
        
        question = data.get('question')
        references = data.get('references')
        if question is None or references is None:
            raise Exception("'question' and 'references' need to be provided.")
        
        return [[question, reference] for reference in references]
        
    
    def inference(self, inputs):
        """
        Compute the relevance scores for the provided inputs.

        Args:
            inputs: list of lists of strings [["question", "reference1"], ["question", "reference2"]]
        Returns:
            numpy array with the relevance scores 
        """
        print(inputs)

        predictions = self.model.predict(inputs)

        logger.info('Predictions successfully computed')
        return predictions

    
    def postprocess(self, outputs):
        """
        Convert the numpy array into a list.

        Args:
            outputs: numpy array containing the relevance scores
        Returns:
            a list containing the relevance scores
        """
        logger.info('Postprocessing successfully computed')
        return [outputs.tolist()]