import torch
from typing import List
import logging
import os
import sentence_transformers
from ts.torch_handler.base_handler import BaseHandler
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)
logger.info("Cross_encoder version %s", sentence_transformers.__version__)


class ModelHandler(BaseHandler):
    def initialize(self, context):
        """
        Initialize function loads the model and the tokenizer

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.

        Raises:
            RuntimeError: Raises the Runtime error when the model or
            tokenizer is missing
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
        Extracts the text from the request

        Args:
            requests: A list containing a dictionary, might be in the form
            of [{'body': json_file}] or [{'data': json_file}] or [{'token_ids': json_file}]
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
        
    
    def inference(self, inputs: List[str]):
        """
        Compute the embeddings given the batch of tokens.

        Args:
            inputs: encoded data
        Returns:
            the tensor containing the batch embeddings.
        """
        print(inputs)

        predictions = self.model.predict(inputs)

        logger.info('Predictions successfully computed')
        return predictions

    
    def postprocess(self, outputs):
        """
        Convert the tensor into a list.

        Args:
            outputs: tensor containing the embeddings.
        Returns:
            the list of list of floating point representing the embeddings for the batch.
        """
        logger.info('Postprocessing successfully computed')
        return [outputs.tolist()]