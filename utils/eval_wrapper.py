# from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
# from mteb.models.model_meta import ModelMeta
# from mteb.models.models_protocols import EncoderProtocol, PromptType
# from .contrastive_datasets import TASK_PROMPTS
# import torch


# def instruction_template(instruction: str, prompt_type: PromptType | None = None) -> str:
#     if not instruction or prompt_type == PromptType.document:
#         return ""
#     if isinstance(instruction, dict):
#         if prompt_type is None:
#             instruction = next(iter(instruction.values()))  # TODO
#         else:
#             instruction = instruction[prompt_type]
#     return f"Instruct: {instruction}\nQuery:"


# def gemma3_instruct_loader(model_name_or_path: str, revision: str, **kwargs) -> EncoderProtocol:
#     model = InstructSentenceTransformerModel(
#         model_name_or_path,
#         revision=revision,
#         instruction_template=instruction_template,
#         apply_instruction_to_passages=False,
#         **kwargs,
#     )
#     return model


# model = ModelMeta(loader=gemma3_instruct_loader, name="my_model", revision="default")


# class InstructSentenceTransformerModel(AbsEncoder):
#     """Instruction wrapper for Sentence Transformer models."""

#     def __init__(
#         self,
#         model_name: str,
#         revision: str,
#         instruction_template: str | Callable[[str, PromptType | None], str] | None = None,
#         max_seq_length: int | None = None,
#         apply_instruction_to_passages: bool = True,
#         padding_side: str | None = None,
#         add_eos_token: bool = False,
#         prompts_dict: dict[str, str] | None = None,
#         **kwargs: Any,
#     ):
#         """Instruct Sentence Transformer Wrapper. Wrapper that passes instructions to the Sentence Transformer model.

#         Applied for models like e5-instruct, jasper, etc.

#         Arguments:
#             model_name: Model name of the sentence transformers model.
#             revision: Revision of the sentence transformers model.
#             instruction_template: Model template. Should contain the string '{instruction}'.
#             max_seq_length: Maximum sequence length. If None, the maximum sequence length will be read from the model config.
#             apply_instruction_to_passages: Whether to apply the instruction template to the passages.
#             padding_side: Padding side. If None, the padding side will be read from the model config.
#             add_eos_token: Whether to add the eos token to each input example.
#             prompts_dict: Dictionary of task names to prompt names. If None, the prompts will be read from the model config.
#             **kwargs: Kwargs for Sentence Transformer model.
#         """

#         self.instruction_template = instruction_template
#         tokenizer_params = {}
#         if add_eos_token:
#             tokenizer_params["add_eos_token"] = add_eos_token
#         if max_seq_length is not None:
#             # https://github.com/UKPLab/sentence-transformers/blob/7341bf155b4349b88690b78c84beb5aa658c439f/sentence_transformers/models/Transformer.py#L115
#             tokenizer_params["model_max_length"] = max_seq_length
#         if padding_side is not None:
#             tokenizer_params["padding_side"] = padding_side

#         kwargs.setdefault("tokenizer_kwargs", {}).update(tokenizer_params)

#         # self.model_name = model_name
#         # self.model = SentenceTransformer(model_name, revision=revision, **kwargs)
#         if max_seq_length:
#             # https://github.com/huggingface/sentence-transformers/issues/3575
#             self.model.max_seq_length = max_seq_length
#         self.apply_instruction_to_passages = apply_instruction_to_passages
#         self.prompts_dict = prompts_dict

#     def encode(
#         self,
#         inputs: DataLoader[BatchedInput],
#         *,
#         task_metadata: TaskMetadata,
#         hf_split: str,
#         hf_subset: str,
#         prompt_type: PromptType | None = None,
#         **kwargs: Any,
#     ) -> Array:
#         """Encodes the given sentences using the encoder.

#         Args:
#             inputs: Batch of inputs to encode.
#             task_metadata: The metadata of the task. Encoders (e.g. SentenceTransformers) use to
#                 select the appropriate prompts, with priority given to more specific task/prompt combinations over general ones.

#                 The order of priorities for prompt selection are:
#                     1. Composed prompt of task name + prompt type (query or passage)
#                     2. Specific task prompt
#                     3. Composed prompt of task type + prompt type (query or passage)
#                     4. Specific task type prompt
#                     5. Specific prompt type (query or passage)
#             hf_split: Split of current task, allows to know some additional information about current split.
#                 E.g. Current language
#             hf_subset: Subset of current task. Similar to `hf_split` to get more information
#             prompt_type: The name type of prompt. (query or passage)
#             **kwargs: Additional arguments to pass to the encoder.

#         Returns:
#             The encoded input in a numpy array or torch tensor of the shape (Number of sentences) x (Embedding dimension).
#         """
#         sentences = [text for batch in inputs for text in batch["text"]]

#         if prompt_type == "query":
#             instruction = TASK_PROMPTS[task_metadata.type]
#         elif prompt_type == "passage":
#             instruction = TASK_PROMPTS["document"]

#         # instruction = self.get_task_instruction(task_metadata, prompt_type)

#         embeddings = self.model.encode(
#             sentences,
#             prompt=instruction,
#             **kwargs,
#         )
#         # check this link on how to encode the sentences consistenlty with sentence transformers
#         # https://github.com/huggingface/sentence-transformers/blob/6aaa53b3c77635b62013080937953ac7c601a894/sentence_transformers/SentenceTransformer.py#L774

#         if isinstance(embeddings, torch.Tensor):
#             # sometimes in kwargs can be return_tensors=True
#             embeddings = embeddings.cpu().detach().float().numpy()
#         return embeddings
