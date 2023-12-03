import torch
import logging
import sentence_transformers 
import os

from ts.torch_handler.base_handler import BaseHandler
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", sentence_transformers.__version__)


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

        # load the model
        model_file = self.manifest['model']['modelFile']
        model_path = os.path.join(model_dir, model_file)

        if os.path.isfile(model_path):
            self.model = SentenceTransformer(model_dir, device=self.device)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f'Successfully loaded model from {model_file}')
        else:
            raise RuntimeError('Missing the model file')

        self.initialized = True


    def preprocess(self, requests):
        """
        Extracts the text from the request

        Args:
            requests: A list containing a dictionary, might be in the form
            of [{'body': json_file}] or [{'data': json_file}]
        Returns:
            the list of strings.
        """

        # unpack the data
        data = requests[0].get('body')
        if data is None:
            data = requests[0].get('data')
        
        texts = data.get('input')
        if texts is not None:
            logger.info('Text provided')
            return texts

        raise Exception("unsupported payload")

    def inference(self, texts):
        """
        Compute the embeddings given the list of strings.

        Args:
            inputs: list of strings
        Returns:
            the tensor containing the batch embeddings.
        """
        # print(inputs)

        text_embeddings = self.model.encode(texts,
                                           normalize_embeddings=True,
                                           convert_to_tensor= True,
                                           device=self.device)

        logger.info('Embeddings successfully computed')
        return text_embeddings
    
    def postprocess(self, outputs: list):
        """
        Convert the tensor into a list.

        Args:
            outputs: tensor containing the embeddings.
        Returns:
            the list of list of floating point representing the embeddings for the batch.
        """
        logger.info('Postprocessing successfully computed')
        return [outputs.tolist()]