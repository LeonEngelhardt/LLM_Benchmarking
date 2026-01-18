def build_prompt(question: str, examples: list[dict] = None, mode: str = "one_shot") -> str:
    prompt = ""
    
    if mode in ["one_shot", "two_shot"]:
        if examples:
            for ex in examples:
                prompt += f"Question: {ex['question']}\nAnswer: {ex['answer']}\nRationale: {ex.get('rationale','')}\n"
                if 'image_path' in ex and ex['image_path']:
                    prompt += f"(Siehe Bild: {ex['image_path']})\n"
                prompt += "\n"
        prompt += f"Question: {question['question']}\n"
        if 'image_path' in question and question['image_path']:
            prompt += f"(See via picture: {question['image_path']})\n"
        prompt += "Answer:"

    elif mode == "learning_from_experience":
        if examples:
            for ex in examples:
                prompt += f"Question: {ex['question']}\n"
                if 'image_path' in ex and ex['image_path']:
                    prompt += f"(See via picture: {ex['image_path']})\n"
        prompt += f"Question: {question['question']}\nAnswer with 'You are correct' or 'You are incorrect':"

    return prompt
