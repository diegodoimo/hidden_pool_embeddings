from .abs_task import AbsTask, TaskMetadata
from .prompts import QWEN3_PROMPTS as TASK_PROMPTS


class MSMARCO(AbsTask):
    """MSMARCO with deduplication against MTEB dev split."""
    hf_name = "mteb/msmarco"
    split = "train"
    has_multiple_datasets = True
    custom_loader = "from_multiple_hf_datasets_with_dedup"
    eval_split = "dev"  # MTEB evaluates on dev split
    anchor_name = "queries"
    positive_name = "corpus"
    qrels_name = "default"
    qrels_fields = {
        "anchor_id": "query-id",
        "positive_id": "corpus-id",
        "score": "score",
    }
    anchor_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text", "title": "title"}
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["MSMARCO"]})


class MSMARCOv2(AbsTask):
    hf_name = "mteb/msmarco-v2"
    split = "train"
    has_multiple_datasets = True
    anchor_name = "queries"
    positive_name = "corpus"
    qrels_name = "default"
    qrels_fields = {
        "anchor_id": "query-id",
        "positive_id": "corpus-id",
        "score": "score",
    }
    anchor_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text", "title": "title"}
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["MSMARCO"]})


class NFCorpus(AbsTask):
    """NFCorpus with deduplication against MTEB test split."""
    hf_name = "mteb/nfcorpus"
    split = "train"
    has_multiple_datasets = True
    custom_loader = "from_multiple_hf_datasets_with_dedup"
    eval_split = "test"  # MTEB evaluates on test split
    anchor_name = "queries"
    positive_name = "corpus"
    qrels_name = "default"
    qrels_fields = {
        "anchor_id": "query-id",
        "positive_id": "corpus-id",
        "score": "score",
    }
    anchor_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text", "title": "title"}
    metadata = TaskMetadata(
        type="Retrieval", prompt={"query": TASK_PROMPTS["NFCorpus"]}
    )


class FEVER(AbsTask):
    """FEVER with deduplication against MTEB test split."""
    hf_name = "mteb/fever"
    split = "train"
    has_multiple_datasets = True
    custom_loader = "from_multiple_hf_datasets_with_dedup"
    eval_split = "test"  # MTEB evaluates on test split
    anchor_name = "queries"
    positive_name = "corpus"
    qrels_name = "default"
    qrels_fields = {
        "anchor_id": "query-id",
        "positive_id": "corpus-id",
        "score": "score",
    }
    anchor_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text", "title": "title"}
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["FEVER"]})


class HotpotQA(AbsTask):
    """HotpotQA with deduplication against MTEB dev split."""
    hf_name = "mteb/hotpotqa"
    split = "train"
    has_multiple_datasets = True
    custom_loader = "from_multiple_hf_datasets_with_dedup"
    eval_split = "dev"  # MTEB evaluates on dev split
    anchor_name = "queries"
    positive_name = "corpus"
    qrels_name = "default"
    qrels_fields = {
        "anchor_id": "query-id",
        "positive_id": "corpus-id",
        "score": "score",
    }
    anchor_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text", "title": "title"}
    metadata = TaskMetadata(
        type="Retrieval", prompt={"query": TASK_PROMPTS["HotpotQA"]}
    )


class NaturalQuestions(AbsTask):
    hf_name = "sentence-transformers/natural-questions"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "query"
    positive_name = "answer"
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["NQ"]})


class ALL_NLI(AbsTask):
    hf_name = "sentence-transformers/all-nli"
    hf_subset = "triplet"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "anchor"
    positive_name = "positive"
    negative_name = "negative"
    metadata = TaskMetadata(
        type="Retrieval", prompt={"query": "Retrieve semantically similar text"}
    )


class Arguana(AbsTask):
    """BeIR Arguana dataset with deduplication against mteb/arguana eval set."""

    hf_name = "BeIR/arguana-generated-queries"
    split = "train"
    has_multiple_datasets = False
    custom_loader = "load_arguana_dedup_retrieval"
    anchor_name = "query"
    positive_name = "text"
    negative_name = "negative"
    metadata = TaskMetadata(
        type="Retrieval", prompt={"query": TASK_PROMPTS["ArguAna"]}
    )


class SNLI(AbsTask):
    """SNLI dataset for retrieval - premise as query, entailed hypothesis as positive."""

    hf_name = "stanfordnlp/snli"
    split = "train"
    has_multiple_datasets = False
    custom_loader = "load_nli_retrieval"
    anchor_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0  # 0 = entailment in SNLI
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )


class MNLI(AbsTask):
    """MNLI dataset for retrieval - premise as query, entailed hypothesis as positive."""

    hf_name = "nyu-mll/multi_nli"
    split = "train"
    has_multiple_datasets = False
    custom_loader = "load_nli_retrieval"
    anchor_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0  # 0 = entailment in MNLI
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )


class ANLI(AbsTask):
    """ANLI dataset for retrieval - premise as query, entailed hypothesis as positive."""

    hf_name = "facebook/anli"
    split = "train_r1"  # Can also use train_r2, train_r3
    has_multiple_datasets = False
    custom_loader = "load_nli_retrieval"
    anchor_name = "premise"
    positive_name = "hypothesis"
    label_name = "label"
    entailment_label = 0  # 0 = entailment in ANLI
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
    )


class PAQ(AbsTask):
    """PAQ (Probably Asked Questions) dataset for retrieval."""

    hf_name = "sentence-transformers/paq"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "query"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve passages that answer the question"
        },
    )


class SQuAD(AbsTask):
    """SQuAD dataset for retrieval - question as query, context as positive."""

    hf_name = "rajpurkar/squad"
    split = "train"
    has_multiple_datasets = False
    custom_loader = "load_squad_retrieval"
    anchor_name = "question"
    positive_name = "context"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve passages that answer the question"
        },
    )


class StackExchangeRetrieval(AbsTask):
    """StackExchange dataset for retrieval - title+body as query, answer as positive."""

    hf_name = "flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl"
    hf_subset = "apple"  # Default subset, can be changed
    split = "train"
    has_multiple_datasets = False
    custom_loader = "load_stackexchange_retrieval"
    anchor_name = "title_body"
    positive_name = "upvoted_answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve answers that best answer the question"
        },
    )


class ELI5(AbsTask):
    """ELI5 (Explain Like I'm 5) dataset for retrieval."""

    hf_name = "sentence-transformers/eli5"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "question"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve passages that answer the question"
        },
    )


class FiQA2018(AbsTask):
    """FiQA 2018 financial QA dataset with deduplication against MTEB test split."""

    hf_name = "mteb/fiqa"
    split = "train"
    has_multiple_datasets = True
    custom_loader = "from_multiple_hf_datasets_with_dedup"
    eval_split = "test"  # MTEB evaluates on test split
    anchor_name = "queries"
    positive_name = "corpus"
    qrels_name = "default"
    qrels_fields = {
        "anchor_id": "query-id",
        "positive_id": "corpus-id",
        "score": "score",
    }
    anchor_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text"}
    metadata = TaskMetadata(
        type="Retrieval", prompt={"query": TASK_PROMPTS["FiQA2018"]}
    )


class BioASQ(AbsTask):
    """BioASQ biomedical QA dataset for retrieval."""

    hf_name = "BeIR/bioasq-generated-queries"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "query"
    positive_name = "text"
    negative_name = "negative"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a biomedical question, retrieve relevant passages that answer the question"
        },
    )


class MIRACL(AbsTask):
    """MIRACL multilingual retrieval dataset.
    
    Note: Each language is a separate config. Set hf_subset to specific language
    (ar, bn, en, es, fa, fi, fr, hi, id, ja, ko, ru, sw, te, th, zh, yo, de)
    or None to load all languages (requires custom loader modification).
    """

    hf_name = "miracl/miracl"
    hf_subset = None  # Load all available languages
    split = "train"
    has_multiple_datasets = False
    custom_loader = "load_miracl_retrieval"
    anchor_name = "query"
    positive_name = "positive_passages"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve relevant passages that answer the question"
        },
    )


class MrTyDi(AbsTask):
    """Mr.TyDi multilingual retrieval with deduplication against MTEB test split."""

    hf_name = "mteb/mrtydi"
    split = "train"
    has_multiple_datasets = True
    custom_loader = "from_multiple_hf_datasets_with_dedup"
    eval_split = "test"  # MTEB evaluates on test split
    anchor_name = "queries"
    positive_name = "corpus"
    qrels_name = "default"
    qrels_fields = {
        "anchor_id": "query-id",
        "positive_id": "corpus-id",
        "score": "score",
    }
    anchor_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text"}
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a question, retrieve relevant passages that answer the question"
        },
    )


class SciFact(AbsTask):
    """SciFact scientific claim verification with deduplication against MTEB test split."""

    hf_name = "mteb/scifact"
    split = "train"
    has_multiple_datasets = True
    custom_loader = "from_multiple_hf_datasets_with_dedup"
    eval_split = "test"  # MTEB evaluates on test split
    anchor_name = "queries"
    positive_name = "corpus"
    qrels_name = "default"
    qrels_fields = {
        "anchor_id": "query-id",
        "positive_id": "corpus-id",
        "score": "score",
    }
    anchor_fields = {"id": "_id", "text": "text"}
    corpus_fields = {"id": "_id", "text": "text", "title": "title"}
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["SciFact"]})


class TriviaQA(AbsTask):
    """TriviaQA dataset for retrieval."""

    hf_name = "sentence-transformers/trivia-qa"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "query"
    positive_name = "answer"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a trivia question, retrieve the answer"},
    )


class COLIEE(AbsTask):
    """COLIEE legal case retrieval dataset."""

    hf_name = "sentence-transformers/coliee"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "anchor"
    positive_name = "positive"
    negative_name = "negative"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a legal case, retrieve relevant legal cases"},
    )


class PubMedQA(AbsTask):
    """PubMedQA biomedical QA dataset for retrieval."""

    hf_name = "qiaojin/PubMedQA"
    hf_subset = "pqa_labeled"
    split = "train"
    has_multiple_datasets = False
    custom_loader = "load_pubmedqa_retrieval"
    anchor_name = "question"
    positive_name = "context"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={
            "query": "Given a biomedical question, retrieve relevant passages that answer the question"
        },
    )


class S2ORCTitleAbstract(AbsTask):
    """S2ORC Title-Abstract retrieval dataset."""

    hf_name = "sentence-transformers/s2orc"
    hf_subset = "title-abstract-pair"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "title"
    positive_name = "abstract"
    metadata = TaskMetadata(
        type="Retrieval", prompt={"query": "Given a paper title, retrieve the abstract"}
    )


class S2ORCTitleCitation(AbsTask):
    """S2ORC Title-Citation retrieval dataset."""

    hf_name = "sentence-transformers/s2orc"
    hf_subset = "title-citation-prediction-triplet"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "anchor"
    positive_name = "positive"
    negative_name = "negative"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a paper title, retrieve titles of cited papers"},
    )


class S2ORCAbstractCitation(AbsTask):
    """S2ORC Abstract-Citation retrieval dataset."""

    hf_name = "sentence-transformers/s2orc"
    hf_subset = "abstract-citation-prediction-triplet"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "anchor"
    positive_name = "positive"
    negative_name = "negative"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a paper abstract, retrieve abstracts of cited papers"},
    )


class SPECTER(AbsTask):
    """SPECTER scientific paper similarity dataset."""

    hf_name = "sentence-transformers/specter"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "anchor"
    positive_name = "positive"
    negative_name = "negative"
    metadata = TaskMetadata(type="Retrieval", prompt={"query": TASK_PROMPTS["SCIDOCS"]})


class XSum(AbsTask):
    """XSum summarization dataset for retrieval (summary -> document)."""

    hf_name = "EdinburghNLP/xsum"
    split = "train"
    has_multiple_datasets = False
    custom_loader = "load_xsum_retrieval"
    anchor_name = "summary"
    positive_name = "document"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a summary, retrieve the original document"},
    )


class CNNDM(AbsTask):
    """CNN/DailyMail summarization dataset for retrieval (highlights -> article)."""

    hf_name = "abisee/cnn_dailymail"
    hf_subset = "3.0.0"
    split = "train"
    has_multiple_datasets = False
    custom_loader = "load_cnndm_retrieval"
    anchor_name = "highlights"
    positive_name = "article"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a summary, retrieve the original article"},
    )


class SentenceCompression(AbsTask):
    """Sentence Compression dataset for retrieval."""

    hf_name = "sentence-transformers/sentence-compression"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "anchor"
    positive_name = "positive"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a compressed sentence, retrieve the original sentence"},
    )


class StackExchangeDupQuestionsS2S(AbsTask):
    """StackExchange duplicate questions (title to title)."""

    hf_name = "sentence-transformers/stackexchange-duplicates"
    hf_subset = "title-title-pair"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "anchor"
    positive_name = "positive"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a question title, retrieve duplicate question titles"},
    )


class StackExchangeDupQuestionsP2P(AbsTask):
    """StackExchange duplicate questions (post to post)."""

    hf_name = "sentence-transformers/stackexchange-duplicates"
    hf_subset = "post-post-pair"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "anchor"
    positive_name = "positive"
    metadata = TaskMetadata(
        type="Retrieval",
        prompt={"query": "Given a question post, retrieve duplicate question posts"},
    )


class QQP(AbsTask):
    """Quora Question Pairs dataset for retrieval."""

    hf_name = "sentence-transformers/quora-duplicates"
    hf_subset = "pair"
    split = "train"
    has_multiple_datasets = False
    anchor_name = "anchor"
    positive_name = "positive"
    metadata = TaskMetadata(
        type="Retrieval", prompt={"query": TASK_PROMPTS["QuoraRetrieval"]}
    )


class StackOverflowDupQuestions(AbsTask):
    """StackOverflow duplicate questions reranking dataset."""

    hf_name = "mteb/stackoverflowdupquestions-reranking"
    split = "train"
    has_multiple_datasets = False
    custom_loader = "load_stackoverflow_dup_retrieval"
    anchor_name = "query"
    positive_name = "positive"
    metadata = TaskMetadata(
        type="Retrieval", prompt={"query": TASK_PROMPTS["StackOverflowDupQuestions"]}
    )



