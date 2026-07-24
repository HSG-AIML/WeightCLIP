import timm
from torch import nn
from huggingface_hub import HfApi, ModelInfo
from huggingface_hub.errors import HfHubHTTPError
from dataclasses import dataclass, field
from typing import List, Optional

from transformers import (
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering
)

from sane.data.datasets.checkpoints_dataset import Checkpoint, CheckpointsDataset



__all__ = [name for name in globals() if not name.startswith('_')]

_task_to_loader = {
    "image-classification": AutoModelForImageClassification,
    "text-classification": AutoModelForSequenceClassification,
    "token-classification": AutoModelForTokenClassification,
    "question-answering": AutoModelForQuestionAnswering
}


@dataclass
class HFQuery:
    """
    Query specification for listing models of Hugging Face using the huggingface_hub API. Each model has a
    modelId - a string identifier - that uniquely specifies a pretrained model hosted in the Hugging Face Model Hub.
    A model can have multiple tags attached to it. Tags are descriptive labels attached to models to categorize
    and annotate them by their characteristics, use cases, or domains.

    Attributes:
        keywords (List[str]): To match modelIds of models on the Hub.
        tags (List[str]): To match models on the Hub possessing these tags.
        limit (int): Maximum number of models to fetch.
        param_limit (int): Maximum number of parameters a model can have.
        require_license (bool): Whether to include models without a specified license.
        allowed_licenses (List[str]): List of acceptable licenses.
        exclude_modelIds (List[str]): List of modelIds to exclude from search.
    """
    keywords: List[str]
    tags: List[str] = field(default_factory=list)
    limit: Optional[int] = None
    param_limit: Optional[int] = None
    require_license: bool = False
    allowed_licenses: Optional[List[str]] = None
    exclude_modelIds: List[str] = field(default_factory=list)


class HFDataset(CheckpointsDataset):
    def __init__(self, query: HFQuery, cache_dir: Optional[str] = None):
        """
        Args:
            cache_dir (str): The location where Hugging Face API caches downloaded checkpoints (defaults to ~/.cache/huggingface).
            query (HFQuery): The query to filter and download models from Hugging Face.
        """
        super().__init__()
        self.cache_dir = cache_dir
        self.model_info_list = []
        self.__fetchModelInfoList(query)
    
    
    def __getitem__(self, index: int) -> Checkpoint:
        model_info = self.model_info_list[index]
        model = self.__loadModel(model_info)
        return Checkpoint(model = model, metadata = model_info.__dict__)
        

    def __len__(self) -> int:
        return len(self.model_info_list)


    def __fetchModelInfoList(self, query: HFQuery):
        """ 
        Given a query this method fetches a list of ModelInfo objects satisfying
        the query criteria and keeps only the ones that can be instantiated.
        
        Args:
            query (HFQuery): The query to filter and download models from Hugging Face.
        """
        if query.allowed_licenses and not query.require_license:
            query.require_license = True

        try:
            api = HfApi()
            pipeline_tag = query.tags[0] if query.tags else None
            matched_models = api.list_models(search=query.keywords, pipeline_tag=pipeline_tag, cardData=True)

            for m in matched_models:
                if m.modelId in query.exclude_modelIds:
                    continue

                card = m.cardData
                if query.require_license:
                    if card is None or getattr(card, 'license', None) is None:
                        continue

                if query.allowed_licenses and getattr(card, 'license', None) not in query.allowed_licenses:
                    continue

                try:
                    model = self.__loadModel(m)

                except HfHubHTTPError as e:
                    if e.response is not None and e.response.status_code == 429:
                        raise RuntimeError("Rate limited: Too Many Requests.")

                except Exception as e: # if model loading fails we discard the model.
                    # TODO: Implement a logger.
                    print(f"skipping {m.modelId} due to error: {e}")
                    continue

                # store necessary information locally.
                total_params = sum(p.numel() for p in model.parameters())
                if query.param_limit and total_params > query.param_limit:
                    continue 

                self.model_info_list.append(m)

                if len(self.model_info_list) >=  query.limit:
                    break

        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                raise RuntimeError("Rate limited: Too Many Requests.")
    

    def __loadModel(self, model_info: ModelInfo) -> nn.Module:
        """
        Uses a ModelInfo instance to download and instantiate a model.
        There are two ways of doing that, ordered by priority:
        (1) Using Timm framework if the modelId suggests that the model originates from Timm.
        (2) Using the HuggingFace AutoModel classes.
        """
        model_id = model_info.modelId

        if model_id.startswith("timm/"): # The model is provided by the Timm organization.
            _, model_name = model_id.split('/') # The second part after '/' corresponds to the model name.
            return timm.create_model(model_name, pretrained=True, features_only=False)
        else:
            # Loads the model using the appropriate AutoModel class from Hugging Face.
            model_task = None
            supported_tasks = _task_to_loader.keys()
            if hasattr(model_info.cardData, 'tasks'):
                task_match = [t for t in supported_tasks if t in model_info.cardData.tasks]
                if task_match:
                    model_task = task_match[0]

            loader = _task_to_loader.get(model_task, AutoModel)
            model = loader.from_pretrained(model_id, cache_dir=self.cache_dir, device_map="cpu")

        return model
