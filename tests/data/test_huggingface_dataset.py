import os
import pytest

from sane.data.datasets.huggingface_dataset import HFQuery, HFDataset
from sane.data.splitter import RandomSplitter



IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

@pytest.mark.slow
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test runs only locally, not in GitHub Actions.")
def test_dataset_methods_and_random_splitter():
    keyword = "resnet"
    tags = ["image-classification"]
    limit = 10
    licenses = [
        "apache-2.0",
        "mit",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc0-1.0",
        "cc-by-4.0",
        "mpl-2.0",
        "lgpl-2.1",
        "lgpl-3.0",
        "artistic-2.0",
        "unlicense",
        "bsl-1.0",
        "isc",
        "postgresql",
        "zlib",
        "ncsa",
    ]

    exclude_ids=["microsoft/resnet-18","google/resnet", "facebook/detr", "microsoft/resnet-152"]

    config = HFQuery(
        keywords = keyword,
        tags = tags,
        limit = limit,
        allowed_licenses = licenses,
        exclude_modelIds = exclude_ids
    )

    ds = HFDataset(config)
    
    # check that 10 models have been fetched.
    assert(len(ds) == 10)

    # check that the exclusion list was respected.
    assert("microsoft/resnet-18" not in [i.metadata['modelId'] for i in ds])

    # check that an item has been properly created.
    m = ds[0]
    assert(m.model)
    assert(m.metadata)

    # check the random splitter on the resulted dataset.
    rs = RandomSplitter()
    train_ds, test_ds = rs(ds, [0.5, 0.5])
    assert(len(train_ds) == len(test_ds))