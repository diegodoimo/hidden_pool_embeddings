

import mteb

def get_instruction(
    self,
    task_metadata: TaskMetadata,
    prompt_type: PromptType | None,
) -> str:
    """Get the instruction/prompt to be used for encoding sentences.

    Args:
        task_metadata: The metadata of the task. Sentence-transformers uses this to
            determine which prompt to use from a specified dictionary.
            The order of priorities for prompt selection are:
                1. Composed prompt of task name + prompt type (query or passage)
                2. Specific task prompt
                3. Composed prompt of task type + prompt type (query or passage)
                4. Specific task type prompt
                5. Specific prompt type (query or passage)
                6. Default prompt from the task definition
        prompt_type: The name type of prompt. (query or passage)

    Returns:
        The instruction/prompt to be used for encoding sentences.
    """
    prompt = task_metadata.prompt
    if self.prompts_dict and task_metadata.name in self.prompts_dict:
        prompt = self.prompts_dict[task_metadata.name]

    if isinstance(prompt, dict) and prompt_type:
        if prompt.get(prompt_type.value):
            return prompt[prompt_type.value]
        logger.warning(
            f"Prompt type '{prompt_type}' not found in task metadata for task '{task_metadata.name}'."
        )
        return ""

    if prompt:
        return prompt

    abstask = mteb.get_task(task_name=task_metadata.name)
    return abstask.abstask_prompt


def format_instruction(
    self, instruction: str, prompt_type: PromptType | None = None
) -> str:
    """Format the instruction using the instruction template.

    Args:
        instruction: The instruction to be formatted.
        prompt_type: The name type of prompt. (query or passage)
    """
    if self.instruction_template is None:
        raise ValueError(
            "Attempting to format an instruction without an instruction template."
        )
    if isinstance(self.instruction_template, str):
        if "{instruction}" not in self.instruction_template:
            raise ValueError(
                "Instruction template must contain the string '{instruction}'."
            )
        return self.instruction_template.format(instruction=instruction)
    return self.instruction_template(instruction, prompt_type)

def get_task_instruction(
    self,
    task_metadata: TaskMetadata,
    prompt_type: PromptType | None,
) -> str:
    """Create the instruction to be used for encoding sentences.

    Args:
        task_metadata: The metadata of the task
        prompt_type: The name type of prompt. (query or passage)

    Returns:
        The instruction to be used for encoding sentences.
    """
    instruction = self.get_instruction(task_metadata, prompt_type)
    if self.instruction_template and len(instruction) > 0:
        return self.format_instruction(instruction, prompt_type)
    return instruction