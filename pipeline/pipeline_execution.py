import torch
from tqdm import tqdm
from judges.llm_as_judge import llm_as_judge
from judges.intention_judge import intention_judge
from collections import Counter
from typing import List
from scanner.scanner import Scanner


def voting(results: List[str]) -> str:
    """
    Determines the majority vote from a list of results.

    Parameters:
        results (List[str]): List of results where each result is 'SAFE' or 'UNSAFE'.

    Returns:
        str: The majority vote, either 'SAFE' or 'UNSAFE'.
    """
    vote_count = Counter(results)
    return 'UNSAFE' if vote_count['UNSAFE'] >= 2 else 'SAFE'


def pipeline_execution(df, saver, columns, target_model, defended_target_model, scanner, moderator, zero_shot_model,
                       substring_scanner, word_scanner, batch_size):
    """
    Executes the pipeline for evaluating prompts using various models and saves the results.

    Parameters:
        df (pd.DataFrame): DataFrame containing the prompts.
        saver (Saver): Object responsible for saving results.
        columns (List[str]): Column names for saving results.
        target_model (TargetLM): The target language model for initial evaluation.
        defended_target_model (DefendedTargetLM): The defended model for secondary evaluation.
        scanner (Scanner): Model used for scanning prompts.
        moderator (Moderator): Model used for moderating prompts.
        zero_shot_model (ZeroShotModel): Model used for zero-shot evaluation.
        substring_scanner (BanSubstrings): Scanner to check for banned substrings.
        word_scanner (BanWords): Scanner to check for banned words.
        batch_size (int): Number of prompts to process per batch.

    Returns:
        None
    """
    for i in tqdm(range(0, len(df), batch_size), desc="Processing"):
        batch = df.iloc[i:i + batch_size]
        attack_prompts = batch['content'].tolist()

        # Iterate over each prompt and apply the scanners
        for index, prompt in enumerate(attack_prompts):
            # Step 1: Apply BanSubstrings and BanWords scanners first
            substrings_result, substrings_confidence = substring_scanner.scan(prompt)
            words_result, words_confidence = word_scanner.scan(prompt)

            if substrings_result == 'UNSAFE' or words_result == 'UNSAFE':
                final_result = 'UNSAFE'
                label = 1  # Unsafe label
                final_response = target_model.get_response(prompt, final_result)
                print(f"Prompt: {prompt}")
                print(f"Final Judgement: {final_result}")
                print(f"Final Response: {final_response}")
                saver.add_and_save(dict(zip(columns, [i + index, prompt, label, final_response])))
                continue  # Skip further model-based evaluations if UNSAFE

            # Step 2: Proceed with other model-based judgments if SAFE
            judgements1 = [llm_as_judge(prompt, target_model)[0] for prompt in attack_prompts]
            judgements2 = [intention_judge(prompt, target_model)[0] for prompt in attack_prompts]
            judgements3 = [scanner.scan(prompt)[0] for prompt in attack_prompts]
            judgements4 = [moderator.moderate([{"role": "user", "content": prompt}])[0] for prompt in attack_prompts]
            judgements5 = [defended_target_model.get_response(prompt)[0] for prompt in attack_prompts]
            judgements6 = [zero_shot_model.scan(prompt)[0] for prompt in attack_prompts]

            # Step 3: Gather results and make a final decision
            # Flatten the list of judgments by selecting only the first element (i.e., the decision) from each result
            results = [judgements1, judgements2, judgements3, judgements4, judgements5, judgements6]
            flattened_results = [judgement[0] for judgement in results]  # Flatten the list

            final_result = voting(flattened_results)

            # Print and save results
            print(f"Prompt: {prompt}")
            print(f"Results: {flattened_results}")
            print(f"Final Judgement: {final_result}")
            label = 1 if final_result == 'UNSAFE' else 0

            final_response = target_model.get_response(prompt, final_result)
            print(f"Final Response: {final_response}")
            saver.add_and_save(dict(zip(columns, [i + index, prompt, label, final_response])))

            # Clear memory
            del final_response
            del results
            del final_result
            torch.cuda.empty_cache()

    saver.final_save()
