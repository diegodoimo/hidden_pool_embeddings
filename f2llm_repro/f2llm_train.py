from tqdm.auto import tqdm

# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import os
import json
from utils.helpers import get_cpt_steps
from utils.create_datasets import get_eval_tasks


class F2LLMEvalWrapper(torch.nn.Module):
    """Thin nn.Module that wraps F2LLM.lm and applies last-token pooling,
    matching the ``(input_ids, attention_mask) -> embeddings`` interface
    expected by ``evaluate_retrieval``."""

    def __init__(self, lm, device):
        super().__init__()
        self._lm = lm
        self._device = device

    @property
    def device(self):
        return self._device

    def forward(self, input_ids, attention_mask):
        outputs = self._lm(input_ids=input_ids, attention_mask=attention_mask)
        # Last non-padding token index for each sequence
        seq_lens = attention_mask.sum(dim=1)  # [B]
        last_idx = seq_lens - 1  # [B]
        embeddings = outputs.last_hidden_state[
            torch.arange(input_ids.size(0), device=input_ids.device),
            last_idx,
        ]  # [B, d]
        # crucial embeddings should be normalized
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class EmbeddingModelEvalWrapper(torch.nn.Module):
    """Thin nn.Module for models that already return L2-normalised embeddings
    directly from their forward pass (e.g. EmbeddingT5Gemma2 which applies
    MeanPooling -> Projection -> Normalize internally).

    This wrapper simply forwards (input_ids, attention_mask) to the module,
    matching the ``(input_ids, attention_mask) -> embeddings`` interface
    expected by ``evaluate_retrieval``.  No additional pooling or normalisation
    is applied here because the module already handles both.
    """

    def __init__(self, module, device):
        super().__init__()
        self._module = module
        self._device = device

    @property
    def device(self):
        return self._device

    def forward(self, input_ids, attention_mask):
        return self._module(input_ids=input_ids, attention_mask=attention_mask)


CLASSIFICATION_DATASETS = [
    "amazon_counterfactual",
    "amazon_polarity",
    "imdb",
    "toxic_conversations",
    "cola",
]
CLUSTERING_DATASETS = [
    "amazon_reviews",
    "banking77",
    "emotion",
    "mtop_intent",
    "mtop_domain",
    "massive_scenario",
    "massive_intent",
    "tweet_sentiment_extraction",
    "arxiv_clustering_p2p",
    "arxiv_clustering_s2s",
    "biorxiv_clustering_p2p",
    "biorxiv_clustering_s2s",
    "medrxiv_clustering_p2p",
    "medrxiv_clustering_s2s",
    "reddit_clustering_p2p",
    "reddit_clustering_s2s",
    "stackexchange_clustering_p2p",
    "stackexchange_clustering_s2s",
    "twentynewsgroups",
]
RETRIEVAL_DATASETS = [
    "arguana",
    "snli",
    "mnli",
    "anli",
    "paq",
    "squad",
    "stackexchange",
    "msmarco",
    "natural_questions",
    "hotpotqa",
    "fever",
    "eli5",
    "fiqa",
    "bioasq",
    "nfcorpus",
    "miracl",
    "mrtidy",
    "scifact",
    "qqp",
    "stackoverflowdupquestions",
    "sts12",
    "sts22",
    "stsbenchmark",
    "amazon_qa",
    "cnn_dm",
    "coliee",
    "paq_part2",
    "pubmedqa",
    "s2orc_abstract_citation",
    "s2orc_title_abstract",
    "s2orc_title_citation",
    "sentence_compression",
    "specter",
    "triviaqa",
    "xsum",
    "stackexchange_part2",
    "stackexchangedupquestions_s2s",
    "stackexchangedupquestions_p2p",
]


def write_tensorboard(summary_writer, log_dict: dict, completed_steps):
    pass
    # for key, value in log_dict.items():
    #     summary_writer.add_scalar(key, value, completed_steps)


def _to_serializable(log_dict):
    return {k: v.item() if hasattr(v, "item") else v for k, v in log_dict.items()}


def save_checkpoint(args, accelerator, model, output_dir, lr_scheduler):
    accelerator.wait_for_everyone()
    accelerator.print(f"Saving checkpoint to {output_dir}")

    if accelerator.is_main_process:
        model.tokenizer.save_pretrained(output_dir)
    unwrapped_model = accelerator.unwrap_model(model.lm)
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model.lm),  # this is required for zero 3
    )
    accelerator.wait_for_everyone()


def inbatch_loss(
    query_embeddings,  # [bs, d]
    context_embeddings,  # [bs, d]
    criterion,
    accelerator,
    temperature=0.05,
):

    bs = query_embeddings.size(0)
    a_norm = F.normalize(query_embeddings, p=2, dim=-1)
    # b_norm = torch.nn.functional.normalize(context_embeddings, p=2, dim=-1)
    b_cross_gpus = accelerator.gather(context_embeddings)  # [bs*process, d]
    # print((context_embeddings - b_cross_gpus[bs * accelerator.process_index : bs * accelerator.process_index+bs]).abs().sum())
    b_norm_cross_gpus = F.normalize(b_cross_gpus, p=2, dim=-1)  # ()

    student_logits = (
        torch.matmul(a_norm, b_norm_cross_gpus.t()) / temperature
    )  # [bs, bs*process]

    labels = (
        torch.arange(bs, device=student_logits.device) + bs * accelerator.process_index
    )
    loss_bs = criterion(student_logits, labels)  # (bs)

    loss = loss_bs.mean()

    return loss


def hard_loss(
    query_embeddings,  # [bs, d]
    context_embeddings,  # [bs, d]
    hard_neg_embeddings,  # [bs, num, d]
    criterion,
    accelerator,
    temperature=0.05,
):

    if hard_neg_embeddings is None:
        return 0.0

    bs = query_embeddings.size(0)
    a_norm = F.normalize(query_embeddings, p=2, dim=-1)

    hard_neg_embeddings = torch.concat(
        [context_embeddings.unsqueeze(1), hard_neg_embeddings], dim=1
    )  # [bs, num_hard+1, d]

    hard_norm = F.normalize(hard_neg_embeddings, p=2, dim=-1)
    logits = (a_norm.unsqueeze(1) * hard_norm).sum(-1) / temperature  # [bs, num_hard+1]

    loss_hard = criterion(
        logits, torch.zeros((bs), dtype=torch.long, device=logits.device)
    ).mean()

    return loss_hard


def inbatch_loss2(
    query_embeddings,  # [bs, d]
    context_embeddings,  # [bs, d]
    doc_ids,  # int32 tensor [bs] — doc IDs of the positive contexts (local process)
    criterion,
    accelerator,
    temperature=0.05,
):
    """In-batch InfoNCE loss with false-negative masking.

    Identical to ``inbatch_loss`` except that, for each query i, every
    context j in the (cross-GPU) denominator whose doc_id equals the
    positive doc_id of query i is masked to -inf before the softmax.
    The true positive (position ``labels[i]``) is always kept.
    """
    bs = query_embeddings.size(0)
    dev = query_embeddings.device
    a_norm = F.normalize(query_embeddings, p=2, dim=-1)
    b_cross_gpus = accelerator.gather(context_embeddings)  # [bs*nproc, d]
    b_norm_cross_gpus = F.normalize(b_cross_gpus, p=2, dim=-1)

    all_doc_ids = accelerator.gather(doc_ids)  # [bs*nproc]

    student_logits = (
        torch.matmul(a_norm, b_norm_cross_gpus.t()) / temperature
    )  # [bs, bs*nproc]

    labels = torch.arange(bs, device=dev) + bs * accelerator.process_index

    # same_id[i, j] = True when all_doc_ids[j] == doc_ids[i]
    same_id = doc_ids[:, None] == all_doc_ids[None, :]  # [bs, bs*nproc]

    # Never mask the true positive
    true_pos = torch.zeros_like(same_id)
    true_pos[torch.arange(bs, device=dev), labels] = True

    student_logits = student_logits.masked_fill(same_id & ~true_pos, float("-inf"))

    loss_bs = criterion(student_logits, labels)
    loss = loss_bs.mean()
    return loss


def hard_loss2(
    query_embeddings,   # [bs, d]
    context_embeddings, # [bs, d]
    hard_neg_embeddings,  # [bs, num, d]
    doc_ids,            # int32 tensor [bs] — positive doc IDs
    hard_neg_doc_ids,   # int32 tensor [bs, num] — hard-negative doc IDs
    criterion,
    accelerator,
    temperature=0.05,
):
    """Hard-negative InfoNCE loss with false-negative masking.

    Identical to ``hard_loss`` except that, for each query i, every hard
    negative whose doc_id equals ``doc_ids[i]`` (the positive doc_id) is
    masked to -inf before the softmax.  The positive (index 0) is always
    kept.
    """
    if hard_neg_embeddings is None:
        return 0.0

    bs = query_embeddings.size(0)
    dev = query_embeddings.device
    a_norm = F.normalize(query_embeddings, p=2, dim=-1)

    hard_neg_embeddings = torch.concat(
        [context_embeddings.unsqueeze(1), hard_neg_embeddings], dim=1
    )  # [bs, num_hard+1, d]

    hard_norm = F.normalize(hard_neg_embeddings, p=2, dim=-1)
    logits = (a_norm.unsqueeze(1) * hard_norm).sum(-1) / temperature  # [bs, num_hard+1]

    if hard_neg_doc_ids is not None:
        # match[i, j] = True when hard_neg_doc_ids[i][j] == doc_ids[i]
        match = hard_neg_doc_ids == doc_ids[:, None]  # [bs, num_hard]
        # Prepend a False column so index 0 (the positive) is never masked
        mask = torch.cat(
            [torch.zeros(bs, 1, dtype=torch.bool, device=dev), match], dim=1
        )  # [bs, num_hard+1]
        logits = logits.masked_fill(mask, float("-inf"))

    loss_hard = criterion(
        logits, torch.zeros(bs, dtype=torch.long, device=dev)
    ).mean()

    return loss_hard


def accelerate_train(
    args,
    accelerator,
    model,
    train_dataloader,
    optimizer,
    lr_scheduler,
    num_train_samples,
    evaluator=None,
    per_device_eval_batch_size=8,
    eval_wrapper_class=None,
):
    if eval_wrapper_class is None:
        eval_wrapper_class = F2LLMEvalWrapper
    accelerator.print(
        "**************************************** Start training ****************************************"
    )
    accelerator.print(f" Num train samples = {num_train_samples}")
    accelerator.print(f" Num epochs = {args.train_epochs}")
    accelerator.print(f" Per device batch size = {args.train_batch_size}")
    accelerator.print(
        f" Global batch size = {args.train_batch_size * accelerator.num_processes}"
    )
    accelerator.print(f" Step per epoch = {len(train_dataloader)}")
    accelerator.print(f" Total training steps = {args.train_steps}")
    accelerator.print(
        "************************************************************************************************"
    )
    filename = (
        "_" + args.out_filename if getattr(args, "out_filename", "") != "" else ""
    )
    global RETRIEVAL_DATASETS, CLASSIFICATION_DATASETS, CLUSTERING_DATASETS
    RETRIEVAL_DATASETS = [
        ds for ds in RETRIEVAL_DATASETS if ds in train_dataloader.loader_dict.keys()
    ]
    CLASSIFICATION_DATASETS = [
        ds
        for ds in CLASSIFICATION_DATASETS
        if ds in train_dataloader.loader_dict.keys()
    ]
    CLUSTERING_DATASETS = [
        ds for ds in CLUSTERING_DATASETS if ds in train_dataloader.loader_dict.keys()
    ]

    # summary_writer = (
    #     SummaryWriter(log_dir=args.tb_dir) if accelerator.is_main_process else None
    # )
    stats = {"train": {}, "valid": {}, "test_perf": {}}
    criterion = CrossEntropyLoss(reduction="none")
    pbar = tqdm(range(args.train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    loss_dict = {
        ds_name: torch.tensor(0.0, device=model.lm.device)
        for ds_name in RETRIEVAL_DATASETS
    }
    loss_hard_dict = {
        ds_name: torch.tensor(0.0, device=model.lm.device)
        for ds_name in train_dataloader.loader_dict.keys()
    }
    count_dict = {
        ds_name: torch.tensor(0, device=model.lm.device)
        for ds_name in RETRIEVAL_DATASETS
    }
    count_hard_dict = {
        ds_name: torch.tensor(0, device=model.lm.device)
        for ds_name in train_dataloader.loader_dict.keys()
    }

    # *************************************************************
    # eval_steps, _ = get_cpt_steps(
    #     int(args.eval_steps), args.max_train_steps, logspace=False
    # )
    # checkpointing_steps, _ = get_cpt_steps(
    #     args.checkpointing_steps, args.max_train_steps, logspace=False
    # )
    # log_steps, _ = get_cpt_steps(
    #     int(args.logging_steps), args.max_train_steps, logspace=False
    # )
    # ***************************************************************

    if args.measure_baselines and evaluator is not None:
        accelerator.print("evaluating baseline performaces")
        model.lm.eval()
        unwrapped_lm = accelerator.unwrap_model(model.lm)
        eval_device = torch.device(f"cuda:{accelerator.local_process_index}")
        eval_wrapper = eval_wrapper_class(unwrapped_lm, eval_device)
        _, summary = evaluator.evaluate(
            eval_wrapper, batch_size=per_device_eval_batch_size
        )
        model.lm.train()
        stats["test_perf"][0] = summary
        if accelerator.is_main_process:
            accelerator.print(f"[Baseline] Step = 0: {summary}")
            with open(
                os.path.join(args.output_dir, f"train_logs{filename}.json"), "w"
            ) as f:
                json.dump(stats, f, indent=4)

    model.lm.train()
    for epoch in range(args.train_epochs):
        accelerator.print(f"*************** Starting epoch {epoch+1} ***************")
        train_dataloader.reset_epoch(epoch)
        for batch in train_dataloader:
            # forward and compute loss
            outputs = model.forward(batch)
            # query/passage features: [bs, d]
            # hard_neg_features: [bs, num_hard_neg, d]

            dataset_name = batch["dataset_name"]
            # model.forward now returns [bs, d] directly — no more .squeeze(1)
            q_feat = outputs["query_passage_features"]
            p_feat = outputs["passage_passage_features"]
            n_feat = outputs["negative_passage_features"]

            if dataset_name in RETRIEVAL_DATASETS:
                loss_hard = hard_loss2(
                    q_feat, p_feat, n_feat,
                    batch.get("positive_doc_ids"),
                    batch.get("negative_doc_ids"),
                    criterion,
                    accelerator,
                )
                loss = inbatch_loss2(
                    q_feat, p_feat,
                    batch.get("positive_doc_ids"),
                    criterion,
                    accelerator,
                )
                count_dict[dataset_name] += 1
                loss_dict[dataset_name] += loss.detach().float()
            else:
                # CLUSTERING / CLASSIFICATION: no doc_id masking
                loss_hard = hard_loss(
                    q_feat, p_feat, n_feat,
                    criterion,
                    accelerator,
                )
                loss = 0.0
            count_hard_dict[dataset_name] += 1
            loss_hard_dict[dataset_name] += loss_hard.detach().float()

            loss_total = loss + loss_hard

            # backward, optimizer, scheduler
            accelerator.backward(loss_total)
            optimizer.step()
            lr_scheduler.step()
            # set_to_none=True avoids a memset; sets grads to None instead of zeroing
            optimizer.zero_grad(set_to_none=True)
            if optimizer.param_groups[0]["lr"] < args.min_lr:
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]["lr"] = args.min_lr

            # log
            completed_steps += 1
            if completed_steps % args.log_interval == 0:
                pbar.update(args.log_interval)

                _per_ds = {}
                for k in loss_dict.keys():
                    count = accelerator.gather(count_dict[k]).sum()
                    if count > 0:
                        _per_ds[f"{k}/training_loss_in_batch"] = (
                            accelerator.gather(loss_dict[k]).sum() / count
                        )
                for k in loss_hard_dict.keys():
                    count = accelerator.gather(count_hard_dict[k]).sum()
                    if count > 0:
                        _per_ds[f"{k}/training_loss_hard"] = (
                            accelerator.gather(loss_hard_dict[k]).sum() / count
                        )
                train_log_dict = {"lr": optimizer.param_groups[0]["lr"]}
                train_log_dict["Avg/retrieval/training_loss_in_batch"] = torch.tensor(
                    [
                        v
                        for k, v in _per_ds.items()
                        if k.split("/")[0] in RETRIEVAL_DATASETS
                        and k.endswith("training_loss_in_batch")
                    ]
                ).mean()
                train_log_dict["Avg/retrieval/training_loss_hard"] = torch.tensor(
                    [
                        v
                        for k, v in _per_ds.items()
                        if k.split("/")[0] in RETRIEVAL_DATASETS
                        and k.endswith("training_loss_hard")
                    ]
                ).mean()
                if not args.only_retrieval:
                    train_log_dict["Avg/classification/training_loss_hard"] = (
                        torch.tensor(
                            [
                                v
                                for k, v in _per_ds.items()
                                if k.split("/")[0] in CLASSIFICATION_DATASETS
                                and k.endswith("training_loss_hard")
                            ]
                        ).mean()
                    )
                    train_log_dict["Avg/clustering/training_loss_hard"] = torch.tensor(
                        [
                            v
                            for k, v in _per_ds.items()
                            if k.split("/")[0] in CLUSTERING_DATASETS
                            and k.endswith("training_loss_hard")
                        ]
                    ).mean()

                train_log_dict["Avg/global/training_loss_in_batch"] = train_log_dict[
                    "Avg/retrieval/training_loss_in_batch"
                ]
                train_log_dict["Avg/global/training_loss_hard"] = torch.tensor(
                    [v for k, v in _per_ds.items() if k.endswith("training_loss_hard")]
                ).mean()

                accelerator.print(f"[Train] Step = {completed_steps}")
                if accelerator.is_main_process:
                    # write_tensorboard(summary_writer, train_log_dict, completed_steps)
                    stats["train"][completed_steps] = _to_serializable(train_log_dict)
                    with open(
                        os.path.join(args.output_dir, f"train_logs{filename}.json"), "w"
                    ) as f:
                        json.dump(stats, f, indent=4)
                loss_dict = {
                    ds_name: torch.tensor(0.0, device=model.lm.device)
                    for ds_name in RETRIEVAL_DATASETS
                }
                loss_hard_dict = {
                    ds_name: torch.tensor(0.0, device=model.lm.device)
                    for ds_name in train_dataloader.loader_dict.keys()
                }
                count_dict = {
                    ds_name: torch.tensor(0, device=model.lm.device)
                    for ds_name in RETRIEVAL_DATASETS
                }
                count_hard_dict = {
                    ds_name: torch.tensor(0, device=model.lm.device)
                    for ds_name in train_dataloader.loader_dict.keys()
                }

            # validation
            if completed_steps % args.validation_interval == 0:
                model.lm.eval()

                unwrapped_lm = accelerator.unwrap_model(model.lm)
                eval_device = torch.device(f"cuda:{accelerator.local_process_index}")

                eval_wrapper = eval_wrapper_class(unwrapped_lm, eval_device)
                _, summary = evaluator.evaluate(
                    eval_wrapper, batch_size=per_device_eval_batch_size
                )
                model.lm.train()
                if accelerator.is_main_process:
                    accelerator.print(f"[MTEB] Step = {completed_steps}: {summary}")
                    stats["test_perf"][completed_steps] = summary
                    with open(
                        os.path.join(args.output_dir, f"train_logs{filename}.json"), "w"
                    ) as f:
                        json.dump(stats, f, indent=4)

            if completed_steps >= args.train_steps:
                break

        # Full MTEB end-of-epoch evaluation (mirrors train.py epoch-end logic)
        if evaluator is not None:
            model.lm.eval()
            unwrapped_lm = accelerator.unwrap_model(model.lm)
            eval_device = torch.device(f"cuda:{accelerator.local_process_index}")
            eval_wrapper = eval_wrapper_class(unwrapped_lm, eval_device)

            # Switch to the full mteb_eng_v2 suite for the end-of-epoch run
            full_eval_tasks = get_eval_tasks("mteb_eng_v2")
            evaluator.update_datasets(full_eval_tasks)
            _, summary = evaluator.evaluate(
                eval_wrapper, batch_size=per_device_eval_batch_size
            )
            model.lm.train()

            if accelerator.is_main_process:
                key = f"mteb_eng_v2_full_epoch_{epoch + 1}"
                accelerator.print(f"[MTEB epoch {epoch + 1}] {summary}")
                stats[key] = summary
                with open(
                    os.path.join(args.output_dir, f"train_logs{filename}.json"), "w"
                ) as f:
                    json.dump(stats, f, indent=4)

            # Restore the original eval set for the next mid-training evals
            restore_tasks = get_eval_tasks(args.eval_set)
            evaluator.update_datasets(restore_tasks)

    # if summary_writer:
    #     summary_writer.close()
