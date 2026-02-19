import pandas as pd

def has_valid_image(path):
    return (
        path is not None and
        isinstance(path, str) and
        path.strip() != "" and
        not pd.isna(path)
    )

def build_prompt_parts(target_row, example_rows, mode="one_shot"):
    blocks = []
    image_counter = 1

    # Instruction
    domain = target_row.get("raw_subject")
    domain = domain if domain and str(domain).strip() else "general knowledge"

    instruction = (
        f"You are an expert in {domain}.\n"
        "Solve the following question step-by-step but reason internally.\n"
        "You must end your response with:\n"
        "Answer: [The Answer Text]\n"
        "Do not output anything after this line."
    )
    #blocks.append({"type": "text", "text": instruction})

    # Examples (only for one/two shot)
    if mode in ["one_shot", "two_shot"]:
        for i, ex in enumerate(example_rows, start=1):
            example_text = (
                f"Example {i}:\n"
                f"Question: {ex['question']}"
            )

            if has_valid_image(ex.get("image_path")):
                example_text += f" (see image {image_counter})"
            
            example_text += f"\nAnswer: {ex['answer']}\nRationale: {ex.get('rationale', '')}"
            blocks.append({"type": "text", "text": example_text})

            if has_valid_image(ex.get("image_path")):
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": ex["image_path"]
                    }
                })
                image_counter += 1

    # Target Question
    target_text = f"Now answer the following question:\nQuestion: {target_row['question']}"
    if has_valid_image(target_row.get("image_path")):
        #target_text += f" (see next image.)"
        target_text += f" (see image {image_counter})"
    
    target_text += "\nAnswer:"
    blocks.append({"type": "text", "text": target_text})

    if has_valid_image(target_row.get("image_path")):
        blocks.append({
            "type": "image",
            "source": {
                "type": "url",
                "url": target_row["image_path"]
            }
        })
        image_counter += 1

    return instruction, blocks



"""import pandas as pd

def has_valid_image(path):
    return (
        path is not None and
        isinstance(path, str) and
        path.strip() != "" and
        not pd.isna(path)
    )

def build_prompt_parts(target_row, example_rows, mode="one_shot"):
    blocks = []

    # Instruction
    domain = target_row.get("domain", "general knowledge")

    instruction = (
        f"You are an expert in {domain}.\n"
        "Solve the following question step-by-step.\n"
        "You must end your response with:\n"
        "Answer: [The Answer Text]\n"
        "Do not write anything after this line."
    )

    blocks.append({"type": "text", "text": instruction})

    # Examples (only for one/two shot)
    if mode in ["one_shot", "two_shot"]:
        for i, ex in enumerate(example_rows, start=1):
            example_text = (
                f"Example {i}:\n"
                f"Question: {ex['question']}\n"
                f"Answer: {ex['answer']}\n"
                f"Rationale: {ex.get('rationale', '')}"
            )

            blocks.append({"type": "text", "text": example_text})

            if has_valid_image(ex.get("image_path")):
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": ex["image_path"]
                    }
                })

    # Target Question
    target_text = (
        "Now answer the following question:\n"
        f"Question: {target_row['question']}\n"
        "Answer:"
    )

    blocks.append({"type": "text", "text": target_text})

    if has_valid_image(target_row.get("image_path")):
        blocks.append({
            "type": "image",
            "source": {
                "type": "url",
                "url": target_row["image_path"]
            }
        })


    return blocks"""