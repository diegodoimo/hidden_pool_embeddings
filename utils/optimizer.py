import math
from torch.optim.lr_scheduler import LambdaLR
import torch


def get_scheduler(
    optimizer,
    max_train_steps,
    lr_min_fact=0,
    warmup_ratio=None,
    warmup_steps=None,
):

    if warmup_steps is None and warmup_ratio is None:
        warmup_steps = 0
    elif warmup_steps is None:
        warmup_steps = int(warmup_ratio * max_train_steps)

    warmup_steps = int(max(warmup_steps, 3))

    scheduler_func = lambda x: min(
        lr_min_fact + (1 - lr_min_fact) * min(x, warmup_steps) / warmup_steps,
        lr_min_fact
        + 0.5
        * (1 - lr_min_fact)
        * (1 + math.cos(max(0, x - warmup_steps) / (max_train_steps - warmup_steps) * math.pi)),
    )
    scheduler = LambdaLR(optimizer, lambda x: scheduler_func(x))
    return scheduler, warmup_steps


def get_scheduler_optimizer(
    model,
    args,
    total_steps,
):
    # "use_orig_parameters" needed for this split of parameters!
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
    )

    num_update_steps_per_epoch = math.ceil(total_steps / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    if args.warmup_steps is not None:
        warmup_steps = args.warmup_steps
    if args.warmup_ratio is not None:
        warmup_steps = int(args.warmup_ratio * args.max_train_steps)
        warmup_steps = min(max(5, warmup_steps), 20)
    if args.warmup_steps is None and args.warmup_ratio is None:
        warmup_steps = 5

    scheduler = lambda x: min(
        args.lr_min_fact + (1 - args.lr_min_fact) * min(x, warmup_steps) / warmup_steps,
        args.lr_min_fact
        + 0.5
        * (1 - args.lr_min_fact)
        * (
            1 + math.cos(max(0, x - warmup_steps) / (args.max_train_steps - warmup_steps) * math.pi)
        ),
    )
    lr_scheduler = LambdaLR(optimizer, lambda x: scheduler(x))

    return optimizer, lr_scheduler
