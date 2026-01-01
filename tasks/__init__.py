from .classification_tasks import *
from .retrieval_tasks import *


NAME_TO_TASK = {
    "msmarco-v2": MSMARCOv2,
    "hotpotqa": HotpotQA,
    "naturalquestions": NaturalQuestions,
    "all_nli": ALL_NLI,
    "fever": FEVER,
    "nfcorpus": NFCorpus,
}


def get_task(name: str):
    if name not in NAME_TO_TASK:
        raise ValueError(f"Unknown task '{name}'. Available tasks: {list(NAME_TO_TASK)}")

    task = NAME_TO_TASK[name]

    return task
