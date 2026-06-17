"""Aggregate train-logs JSONs under f2llm_repro/results/train into LaTeX tables."""
import json, os, glob

ROOT = "f2llm_repro/results/train"

TASKS = ["Retrieval", "Reranking", "STS", "Summarization",
         "PairClassification", "Classification", "Clustering"]
AVGS = ["micro_average", "macro_average"]

def load(p):
    with open(p) as fh:
        return json.load(fh)

def last_train_loss(d):
    train = d.get("train") or {}
    if not train:
        return None, (None, None)
    last = max(train.keys(), key=int)
    e = train[last]
    return last, (e.get("Avg/global/training_loss_in_batch"),
                  e.get("Avg/global/training_loss_hard"))

def best_eval(d):
    """Return dict of per-task scores from the most informative source available.
    Priority: mteb_eng_v2_full_epoch_1 > mteb_eng_v2_epoch_1 > latest test_perf step."""
    for k in ("mteb_eng_v2_full_epoch_1", "mteb_eng_v2_epoch_1"):
        if d.get(k):
            return k, d[k]
    tp = d.get("test_perf") or {}
    if tp:
        last = max(tp.keys(), key=int)
        return f"test_perf@{last}", tp[last]
    return None, {}

def fmt(v):
    return f"{v*100:.2f}" if isinstance(v, (int, float)) else "--"

def fmt_loss(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "--"

# ---------------------------------------------------------------
# Collect runs
# ---------------------------------------------------------------

def list_logs(folder, recurse_one_level=False):
    """Return list of (path_to_log, run_dirname_or_None)."""
    out = []
    for f in sorted(glob.glob(os.path.join(folder, "train_logs_*.json"))):
        out.append((f, None))
    if recurse_one_level:
        for sub in sorted(glob.glob(os.path.join(folder, "*"))):
            if os.path.isdir(sub) and os.path.basename(sub) not in {"old_runs"}:
                for f in sorted(glob.glob(os.path.join(sub, "train_logs_*.json"))):
                    out.append((f, os.path.basename(sub)))
    return out

end2end_files = [p for p, _ in list_logs(ROOT) if "f2llm_eval_prompts" not in p]
stage1_files = list_logs(os.path.join(ROOT, "stage1"), recurse_one_level=True)
stage2_files = list_logs(os.path.join(ROOT, "stage2"), recurse_one_level=True)
ts_single_files = list_logs(os.path.join(ROOT, "task_specific"), recurse_one_level=True)
ts_single = [(p, d) for p, d in ts_single_files if d and not d.startswith("stage")]
ts_stage1 = [(p, d) for p, d in ts_single_files if d and "stage1" in p]
ts_stage2 = [(p, d) for p, d in ts_single_files if d and "stage2" in p]

# ---------------------------------------------------------------
# TABLE 1: End-to-end full-data runs
# ---------------------------------------------------------------
# Manual labels (run filename -> short description for hyperparam table)
end2end_meta = {
    "train_logs_deepspeed_t5gemma2_gpus4_bs320_lr1e-05_wd0.01.json":
        ("E1", "plain backbone, EOS pool (baseline)"),
    "train_logs_deepspeed_t5gemma2_gated_attn_h4_cls_query_gpus4_bs320_lr1e-05_wd0.01.json":
        ("E2", "gated attn pool, $h{=}4$, CLS query"),
    "train_logs_deepspeed_t5gemma2_attn_pooling_h4_cls_query_gpus4_bs320_lr1e-05_wd0.01.json":
        ("E3", "attn pool, $h{=}4$, CLS query"),
    "train_logs_deepspeed_t5gemma2_attn_pooling_h1_frozen_gpus4_bs320_lr1e-05_wd0.01.json":
        ("E4", "attn pool $h{=}1$, frozen backbone"),
    "train_logs_deepspeed_t5gemma2_cross_attn_L3_8_13_18_23_gpus4_bs320_lr1e-05_wd0.01.json":
        ("E5", "cross-attn pool, layers 3/8/13/18/23"),
    "train_logs_deepspeed_t5gemma2_lora_cls_attn_r64_gpus4_bs320_lr1e-05_wd0.01.json":
        ("E6", "LoRA CLS attn, $r{=}64$"),
}

# ---------------------------------------------------------------
# TABLE 2: Two-stage full-data runs
# Pair each stage-2 run with its stage-1 sibling via the directory naming.
# ---------------------------------------------------------------
stage1_lookup = {os.path.basename(os.path.dirname(p)): p for p, _ in stage1_files if _}
two_stage_meta = [
    # (s2_dirname,                                                                                              s1_dirname,                                                                  ID, description)
    ("t5gemma2_two_stage_s2_frac1_gated_attn_h4_cls_s1_frac1_gated_attn_h4_mean_lr0.0001_wd0.0_gpus4_bs320_lr1e-05_wd0.01",
        "t5gemma2_two_stage_s1_frac1_gated_attn_h4_mean_gpus4_bs320_lr0.0001_wd0.0",
        "T1", "gated attn $h{=}4$: S1 mean (lr$10^{-4}$/wd$0$) $\\to$ S2 CLS (lr$10^{-5}$/wd$0.01$)"),
    ("t5gemma2_two_stage_s2_frac1_gated_attn_h4_cls_gpus4_bs320_lr2e-05_wd0.01",
        "t5gemma2_two_stage_s1_frac1_gated_attn_h4_mean_gpus4_bs320_lr0.0001_wd0.0",
        "T2", "gated attn $h{=}4$: S1 mean (lr$10^{-4}$/wd$0$) $\\to$ S2 CLS (lr$2{\\cdot}10^{-5}$/wd$0.01$)"),
    ("t5gemma2_two_stage_s2_frac1_gated_attn_h4_mean_s1_frac1_gated_attn_h4_mean_lr0.0001_wd0.0_gpus4_bs320_lr2e-05_wd0.01",
        "t5gemma2_two_stage_s1_frac1_gated_attn_h4_mean_gpus4_bs320_lr0.0001_wd0.0",
        "T3", "gated attn $h{=}4$: S1 mean $\\to$ S2 mean (lr$2{\\cdot}10^{-5}$/wd$0.01$)"),
    ("t5gemma2_two_stage_s2_frac1_attn_pooling_h4_cls_s1_frac1_attn_pooling_h4_mean_lr0.0001_wd0.0_gpus4_bs320_lr2e-05_wd0.01",
        "t5gemma2_two_stage_s1_frac1_attn_pooling_h4_mean_gpus4_bs320_lr0.0001_wd0.0",
        "T4", "ungated attn $h{=}4$: S1 mean $\\to$ S2 CLS (lr$2{\\cdot}10^{-5}$/wd$0.01$)"),
    ("t5gemma2_two_stage_s2_frac1_gated_attn_h8_cls_s1_frac1_gated_attn_h8_mean_lr0.0001_wd0.0_gpus4_bs320_lr2e-05_wd0.01",
        "t5gemma2_two_stage_s1_frac1_gated_attn_h8_mean_gpus4_bs320_lr0.0001_wd0.0",
        "T5", "gated attn $h{=}8$: S1 mean $\\to$ S2 CLS (lr$2{\\cdot}10^{-5}$/wd$0.01$)"),
]

# ---------------------------------------------------------------
# TABLE 3: Task-specific runs
# rows = variant, columns = task
# ---------------------------------------------------------------
ts_variants = [
    # (id,  description,                            dirname template,                                                                    log filename template)
    ("S1", "plain backbone (EOS), single-stage",
        "t5gemma2_task_{T}_gpus4_bs320_lr2e-05_wd0.01",
        "train_logs_t5gemma2_task_{T}_gpus4_bs320_lr2e-05_wd0.01.json",
        "task_specific"),
    ("S2", "gated attn pool $h{=}4$, mean, single-stage",
        "t5gemma2_gated_attn_h4_mean_task_{T}_gpus4_bs320_lr2e-05_wd0.01",
        "train_logs_t5gemma2_gated_attn_h4_mean_task_{T}_gpus4_bs320_lr2e-05_wd0.01.json",
        "task_specific"),
    ("S3", "gated attn pool $h{=}4$, CLS, single-stage",
        "t5gemma2_gated_attn_h4_cls_task_{T}_gpus4_bs320_lr2e-05_wd0.01",
        "train_logs_t5gemma2_gated_attn_h4_cls_task_{T}_gpus4_bs320_lr2e-05_wd0.01.json",
        "task_specific"),
    ("S4", "gated attn $h{=}4$, 2-stage: S1 mean $\\to$ S2 CLS",
        "t5gemma2_two_stage_s2_frac1_gated_attn_h4_cls_task_{T}_gpus4_bs320_lr2e-05_wd0.01",
        "train_logs_t5gemma2_two_stage_s2_frac1_gated_attn_h4_cls_task_{T}_gpus4_bs320_lr2e-05_wd0.01.json",
        "task_specific/stage2"),
]

# ---------------------------------------------------------------
# Render
# ---------------------------------------------------------------
out = []

def latex_header(caption, label, columns):
    s  = "\\begin{table}[ht]\n\\centering\n\\small\n"
    s += "\\setlength{\\tabcolsep}{4pt}\n"
    s += f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
    s += "\\begin{tabular}{" + columns + "}\n\\toprule"
    return s

def latex_footer():
    return "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

# ============================================================
# Section: end-to-end full
# ============================================================
out.append("% =============================================================")
out.append("% End-to-end full-data runs (submit_train_f2llm_torchrun, full data)")
out.append("% =============================================================\n")

# hyperparam table
out.append(latex_header(
    "End-to-end full-data runs: hyperparameter setup. All runs share "
    "backbone \\texttt{t5gemma-2-270m-270m}, $4{\\times}$H100 GPUs, global batch $320$, "
    "max seq.\\ length $1024$, $7$ hard negatives, $1$ epoch, lr $10^{-5}$, weight decay $0.01$.",
    "tab:e2e-hparams",
    "lll"))
out.append("ID & Configuration & Train loss (final) \\\\\n\\midrule")
for fname, (rid, desc) in end2end_meta.items():
    p = os.path.join(ROOT, fname)
    d = load(p)
    last, (lb, lh) = last_train_loss(d)
    train_loss = f"in-batch {fmt_loss(lb)} / hard {fmt_loss(lh)} @ step {last}"
    out.append(f"{rid} & {desc} & {train_loss} \\\\")
out.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# perf table
out.append(latex_header(
    "End-to-end full-data runs: MTEB-eng-v2 task accuracies (\\%) at end of epoch 1.",
    "tab:e2e-perf",
    "l" + "r"*len(TASKS) + "rr"))
out.append("ID & " + " & ".join(TASKS) + " & micro & macro \\\\\n\\midrule")
for fname, (rid, _) in end2end_meta.items():
    p = os.path.join(ROOT, fname)
    d = load(p)
    src, ev = best_eval(d)
    cells = [fmt(ev.get(t)) for t in TASKS] + [fmt(ev.get("micro_average")), fmt(ev.get("macro_average"))]
    out.append(f"{rid} & " + " & ".join(cells) + " \\\\")
out.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ============================================================
# Section: 2-stage full
# ============================================================
out.append("\n% =============================================================")
out.append("% Two-stage full-data runs (submit_train_2stage_f2llm_torchrun)")
out.append("% =============================================================\n")

out.append(latex_header(
    "Two-stage full-data runs: hyperparameter setup and stage-1 final loss. "
    "Stage 1 trains the pooling head with backbone frozen; stage 2 finetunes end-to-end "
    "from the stage-1 head. All runs: $4{\\times}$H100, batch $320$, max seq.\\ $1024$, "
    "$7$ hard negatives, $1$ epoch, S1 fraction $1.0$.",
    "tab:s2-hparams",
    "ll l l"))
out.append("ID & Configuration & S1 loss (final) & S2 loss (final) \\\\\n\\midrule")
for s2_dir, s1_dir, rid, desc in two_stage_meta:
    s2_log = os.path.join(ROOT, "stage2", s2_dir, f"train_logs_{s2_dir}.json")
    s1_log = os.path.join(ROOT, "stage1", s1_dir, f"train_logs_{s1_dir}.json")
    s2 = load(s2_log) if os.path.exists(s2_log) else {}
    s1 = load(s1_log) if os.path.exists(s1_log) else {}
    _, (s1_lb, s1_lh) = last_train_loss(s1) if s1 else (None, (None, None))
    _, (s2_lb, s2_lh) = last_train_loss(s2) if s2 else (None, (None, None))
    out.append(f"{rid} & {desc} & in-batch {fmt_loss(s1_lb)} / hard {fmt_loss(s1_lh)} "
               f"& in-batch {fmt_loss(s2_lb)} / hard {fmt_loss(s2_lh)} \\\\")
out.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

out.append(latex_header(
    "Two-stage full-data runs: MTEB-eng-v2 task accuracies (\\%) at end of stage 2 epoch 1.",
    "tab:s2-perf",
    "l" + "r"*len(TASKS) + "rr"))
out.append("ID & " + " & ".join(TASKS) + " & micro & macro \\\\\n\\midrule")
for s2_dir, s1_dir, rid, desc in two_stage_meta:
    s2_log = os.path.join(ROOT, "stage2", s2_dir, f"train_logs_{s2_dir}.json")
    if not os.path.exists(s2_log):
        cells = ["--"]*(len(TASKS)+2)
    else:
        d = load(s2_log)
        src, ev = best_eval(d)
        cells = [fmt(ev.get(t)) for t in TASKS] + [fmt(ev.get("micro_average")), fmt(ev.get("macro_average"))]
    out.append(f"{rid} & " + " & ".join(cells) + " \\\\")
out.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ============================================================
# Section: task-specific
# ============================================================
out.append("\n% =============================================================")
out.append("% Task-specific runs (submit_train_f2llm_torchrun with TASK_TYPES,")
out.append("% and submit_train_2stage_single_task_torchrun)")
out.append("% =============================================================\n")

ts_tasks = ["Retrieval", "Reranking", "STS", "Summarization",
            "PairClassification", "Classification", "Clustering"]

out.append(latex_header(
    "Task-specific runs: variants. All runs share batch $320$, max seq.\\ $1024$, "
    "$1$ epoch, lr $2{\\cdot}10^{-5}$, wd $0.01$. Stage-1 (for 2-stage) uses lr $10^{-4}$, wd $0$.",
    "tab:ts-hparams",
    "ll"))
out.append("ID & Configuration \\\\\n\\midrule")
for rid, desc, *_ in ts_variants:
    out.append(f"{rid} & {desc} \\\\")
out.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# task-specific perf: rows = variant, cols = task (each cell = accuracy on that task)
out.append(latex_header(
    "Task-specific runs: per-task MTEB accuracy (\\%) at end of epoch 1. "
    "Each cell is the score for a model finetuned only on that task's data.",
    "tab:ts-perf",
    "l" + "r"*len(ts_tasks)))
out.append("ID & " + " & ".join(ts_tasks) + " \\\\\n\\midrule")
for rid, desc, dir_t, log_t, sub in ts_variants:
    cells = []
    for T in ts_tasks:
        path = os.path.join(ROOT, sub, dir_t.format(T=T), log_t.format(T=T))
        if not os.path.exists(path):
            cells.append("--")
            continue
        d = load(path)
        _, ev = best_eval(d)
        # for task-specific runs the trained-task score is keyed by the task name
        # (mteb_eng_v2_epoch_1 contains only that one task)
        v = ev.get(T)
        cells.append(fmt(v))
    out.append(f"{rid} & " + " & ".join(cells) + " \\\\")
out.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# task-specific training-loss companion
out.append(latex_header(
    "Task-specific runs: final reported training loss (Avg/global, in-batch / hard) at the last logged step.",
    "tab:ts-train-loss",
    "l" + "l"*len(ts_tasks)))
out.append("ID & " + " & ".join(ts_tasks) + " \\\\\n\\midrule")
for rid, desc, dir_t, log_t, sub in ts_variants:
    cells = []
    for T in ts_tasks:
        path = os.path.join(ROOT, sub, dir_t.format(T=T), log_t.format(T=T))
        if not os.path.exists(path):
            cells.append("--")
            continue
        d = load(path)
        last, (lb, lh) = last_train_loss(d)
        if last is None:
            cells.append("--")
        else:
            cells.append(f"{fmt_loss(lb)}/{fmt_loss(lh)}")
    out.append(f"{rid} & " + " & ".join(cells) + " \\\\")
out.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

print("\n".join(out))
