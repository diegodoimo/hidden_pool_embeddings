from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.nli_tasks.nli_loaders import load_nli_retrieval


class XNLIAr(AbsTask):
    """XNLI Arabic natural language inference dataset."""

    language = "ar"
    hf_name = "mteb/xnli"
    hf_subset = "ar"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )
    loader = load_nli_retrieval


class XNLIBg(AbsTask):
    """XNLI Bulgarian natural language inference dataset."""

    language = "bg"
    hf_name = "mteb/xnli"
    hf_subset = "bg"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )
    loader = load_nli_retrieval


class XNLIDe(AbsTask):
    """XNLI German natural language inference dataset."""

    language = "de"
    hf_name = "mteb/xnli"
    hf_subset = "de"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )
    loader = load_nli_retrieval


class XNLIEl(AbsTask):
    """XNLI Greek natural language inference dataset."""

    language = "el"
    hf_name = "mteb/xnli"
    hf_subset = "el"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )
    loader = load_nli_retrieval


class XNLIEs(AbsTask):
    """XNLI Spanish natural language inference dataset."""

    language = "es"
    hf_name = "mteb/xnli"
    hf_subset = "es"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )
    loader = load_nli_retrieval


class XNLIFr(AbsTask):
    """XNLI French natural language inference dataset."""

    language = "fr"
    hf_name = "mteb/xnli"
    hf_subset = "fr"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )
    loader = load_nli_retrieval


class XNLIHi(AbsTask):
    """XNLI Hindi natural language inference dataset."""

    language = "hi"
    hf_name = "mteb/xnli"
    hf_subset = "hi"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )
    loader = load_nli_retrieval


class XNLIRu(AbsTask):
    """XNLI Russian natural language inference dataset."""

    language = "ru"
    hf_name = "mteb/xnli"
    hf_subset = "ru"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )
    loader = load_nli_retrieval


class XNLISw(AbsTask):
    """XNLI Swahili natural language inference dataset."""

    language = "sw"
    hf_name = "mteb/xnli"
    hf_subset = "sw"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )
    loader = load_nli_retrieval


class XNLITh(AbsTask):
    """XNLI Thai natural language inference dataset."""

    language = "th"
    hf_name = "mteb/xnli"
    hf_subset = "th"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )
    loader = load_nli_retrieval


class XNLITr(AbsTask):
    """XNLI Turkish natural language inference dataset."""

    language = "tr"
    hf_name = "mteb/xnli"
    hf_subset = "tr"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )
    loader = load_nli_retrieval


class XNLIUr(AbsTask):
    """XNLI Urdu natural language inference dataset."""

    language = "ur"
    hf_name = "mteb/xnli"
    hf_subset = "ur"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )
    loader = load_nli_retrieval


class XNLIVi(AbsTask):
    """XNLI Vietnamese natural language inference dataset."""

    language = "vi"
    hf_name = "mteb/xnli"
    hf_subset = "vi"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )
    loader = load_nli_retrieval


class XNLIZh(AbsTask):
    """XNLI Chinese natural language inference dataset."""

    language = "zh"
    hf_name = "mteb/xnli"
    hf_subset = "zh"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )
    loader = load_nli_retrieval
