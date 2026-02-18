from tasks.abs_task import AbsTask, TaskMetadata
from tasks.prompts import QWEN3_PROMPTS as TASK_PROMPTS
from tasks.retrieval_tasks.retrieval_loaders import from_multiple_hf_datasets


class MrTyDiArabic(AbsTask):
    """Mr.TyDi Arabic retrieval with deduplication against MTEB test split."""

    language = "ar"
    hf_name = "mteb/mrtydi"
    split = "train"
    has_multiple_datasets = True
    eval_split = "test"
    query_name = "arabic-queries"
    positive_name = "arabic-corpus"
    qrels_name = "arabic-qrels"
    qrels_fields = {"query_id": "query-id", "positive_id": "corpus-id", "score": "score"}
    query_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text"}
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a question, retrieve relevant passages that answer the question"},
    )
    loader = from_multiple_hf_datasets


class MrTyDiBengali(AbsTask):
    """Mr.TyDi Bengali retrieval with deduplication against MTEB test split."""

    language = "bn"
    hf_name = "mteb/mrtydi"
    split = "train"
    has_multiple_datasets = True
    eval_split = "test"
    query_name = "bengali-queries"
    positive_name = "bengali-corpus"
    qrels_name = "bengali-qrels"
    qrels_fields = {"query_id": "query-id", "positive_id": "corpus-id", "score": "score"}
    query_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text"}
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a question, retrieve relevant passages that answer the question"},
    )
    loader = from_multiple_hf_datasets


class MrTyDiEnglish(AbsTask):
    """Mr.TyDi English retrieval with deduplication against MTEB test split."""

    language = "en"
    hf_name = "mteb/mrtydi"
    split = "train"
    has_multiple_datasets = True
    eval_split = "test"
    query_name = "english-queries"
    positive_name = "english-corpus"
    qrels_name = "english-qrels"
    qrels_fields = {"query_id": "query-id", "positive_id": "corpus-id", "score": "score"}
    query_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text"}
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a question, retrieve relevant passages that answer the question"},
    )
    loader = from_multiple_hf_datasets


class MrTyDiFinnish(AbsTask):
    """Mr.TyDi Finnish retrieval with deduplication against MTEB test split."""

    language = "fi"
    hf_name = "mteb/mrtydi"
    split = "train"
    has_multiple_datasets = True
    eval_split = "test"
    query_name = "finnish-queries"
    positive_name = "finnish-corpus"
    qrels_name = "finnish-qrels"
    qrels_fields = {"query_id": "query-id", "positive_id": "corpus-id", "score": "score"}
    query_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text"}
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a question, retrieve relevant passages that answer the question"},
    )
    loader = from_multiple_hf_datasets


class MrTyDiIndonesian(AbsTask):
    """Mr.TyDi Indonesian retrieval with deduplication against MTEB test split."""

    language = "id"
    hf_name = "mteb/mrtydi"
    split = "train"
    has_multiple_datasets = True
    eval_split = "test"
    query_name = "indonesian-queries"
    positive_name = "indonesian-corpus"
    qrels_name = "indonesian-qrels"
    qrels_fields = {"query_id": "query-id", "positive_id": "corpus-id", "score": "score"}
    query_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text"}
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a question, retrieve relevant passages that answer the question"},
    )
    loader = from_multiple_hf_datasets


class MrTyDiJapanese(AbsTask):
    """Mr.TyDi Japanese retrieval with deduplication against MTEB test split."""

    language = "ja"
    hf_name = "mteb/mrtydi"
    split = "train"
    has_multiple_datasets = True
    eval_split = "test"
    query_name = "japanese-queries"
    positive_name = "japanese-corpus"
    qrels_name = "japanese-qrels"
    qrels_fields = {"query_id": "query-id", "positive_id": "corpus-id", "score": "score"}
    query_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text"}
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a question, retrieve relevant passages that answer the question"},
    )
    loader = from_multiple_hf_datasets


class MrTyDiKorean(AbsTask):
    """Mr.TyDi Korean retrieval with deduplication against MTEB test split."""

    language = "ko"
    hf_name = "mteb/mrtydi"
    split = "train"
    has_multiple_datasets = True
    eval_split = "test"
    query_name = "korean-queries"
    positive_name = "korean-corpus"
    qrels_name = "korean-qrels"
    qrels_fields = {"query_id": "query-id", "positive_id": "corpus-id", "score": "score"}
    query_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text"}
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a question, retrieve relevant passages that answer the question"},
    )
    loader = from_multiple_hf_datasets


class MrTyDiRussian(AbsTask):
    """Mr.TyDi Russian retrieval with deduplication against MTEB test split."""

    language = "ru"
    hf_name = "mteb/mrtydi"
    split = "train"
    has_multiple_datasets = True
    eval_split = "test"
    query_name = "russian-queries"
    positive_name = "russian-corpus"
    qrels_name = "russian-qrels"
    qrels_fields = {"query_id": "query-id", "positive_id": "corpus-id", "score": "score"}
    query_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text"}
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a question, retrieve relevant passages that answer the question"},
    )
    loader = from_multiple_hf_datasets


class MrTyDiSwahili(AbsTask):
    """Mr.TyDi Swahili retrieval with deduplication against MTEB test split."""

    language = "sw"
    hf_name = "mteb/mrtydi"
    split = "train"
    has_multiple_datasets = True
    eval_split = "test"
    query_name = "swahili-queries"
    positive_name = "swahili-corpus"
    qrels_name = "swahili-qrels"
    qrels_fields = {"query_id": "query-id", "positive_id": "corpus-id", "score": "score"}
    query_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text"}
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a question, retrieve relevant passages that answer the question"},
    )
    loader = from_multiple_hf_datasets


class MrTyDiTelugu(AbsTask):
    """Mr.TyDi Telugu retrieval with deduplication against MTEB test split."""

    language = "te"
    hf_name = "mteb/mrtydi"
    split = "train"
    has_multiple_datasets = True
    eval_split = "test"
    query_name = "telugu-queries"
    positive_name = "telugu-corpus"
    qrels_name = "telugu-qrels"
    qrels_fields = {"query_id": "query-id", "positive_id": "corpus-id", "score": "score"}
    query_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text"}
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a question, retrieve relevant passages that answer the question"},
    )
    loader = from_multiple_hf_datasets


class MrTyDiThai(AbsTask):
    """Mr.TyDi Thai retrieval with deduplication against MTEB test split."""

    language = "th"
    hf_name = "mteb/mrtydi"
    split = "train"
    has_multiple_datasets = True
    eval_split = "test"
    query_name = "thai-queries"
    positive_name = "thai-corpus"
    qrels_name = "thai-qrels"
    qrels_fields = {"query_id": "query-id", "positive_id": "corpus-id", "score": "score"}
    query_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text"}
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a question, retrieve relevant passages that answer the question"},
    )
    loader = from_multiple_hf_datasets
