def build_prompt(target_row: dict, example_rows: list[dict] = None, mode: str = "one_shot", vision: bool = False) -> str:
    category = target_row.get('raw_subject', 'General Knowledge')

    instruction = (
        f"You are an expert in {category}. \n"
        "Solve the following question. You may think step-by-step to derive the answer.\n"
        "You must end your response with the exact phrase:\n"
        "Answer: [The Answer Text]\n"
        "Do not write anything after this line."
    )

    prompt = instruction + "\n\n"

    if example_rows:
        prompt += "Here are example questions with their answers and rationales.\n\n"
        for i, ex in enumerate(example_rows, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Question: {ex['question']}\n"

            if vision and isinstance(ex.get("image_path"), str) and ex["image_path"].strip():
                prompt += f"Image: {ex['image_path']}\n"

            prompt += f"Answer: {ex['answer']}\n"
            if ex.get("rationale"):
                prompt += f"Rationale: {ex['rationale']}\n"

            prompt += "\n"

    if mode in ["one_shot", "two_shot"]:
        prompt += "Now answer the following question:\n"
        prompt += f"Question: {target_row['question']}\n"

    elif mode == "zero_shot":
        base_prompt = f"{instruction}\n\nQuestion: {target_row['question']}\n"
        
        if vision:
            return (
                f"Question: {target_row['question']}\n"
                f"Image: {target_row['image_path']}\n"
                f"Answer:"
            )
        else:
            return (
                f"Question: {target_row['question']}\n"
                f"Answer:"
            )

    if vision and isinstance(target_row.get("image_path"), str) and target_row["image_path"].strip():
        prompt += f"Image: {target_row['image_path']}\n"

    if mode in ["one_shot", "two_shot"]:
        prompt += "Answer:"

    return prompt.strip()