import pandas as pd
from llm.models import TargetLM, DefendedTargetLM
from scanner.injection_scanner import PromptInjectionScanner
from scanner.moderate_scanner import Moderator
from scanner.zero_shot_scanner import ZeroShot
from scanner.string_scanner import BanSubstrings
from scanner.word_scanner import BanWords
from utils.saver import IntervalSaver
from defenses.backtranslation import BacktranslationConfig
from defenses.defense import load_defense


def pipeline_initialization(args):
    """
    Initializes the pipeline for processing prompts with various models and configurations.

    Parameters:
        args (Namespace): Command line arguments or configuration containing settings for initialization.

    Returns:
        dict: A dictionary containing initialized components including models, data, and configurations.
    """
    # Initialize batch size and columns for saving results
    batch_size = args.batch_size
    columns = ['id', 'content', 'label', 'response']

    # Set up the saver for saving results at specified intervals
    saver = IntervalSaver(output_path=args.output_dir, interval=args.save_interval, columns=columns)

    # Load dataset (encoding is set to 'gbk' to handle specific character sets)
    df = pd.read_csv(args.dataset_dir, encoding='gbk')

    substrings = []  # List of substrings to be checked using BanSubstrings scanner
    words = []  # List of words to be checked using BanWords scanner

    # Initialize scanners and models
    substring_scanner = BanSubstrings(substrings)  # Scanner for banned substrings
    word_scanner = BanWords(words)  # Scanner for banned words
    target_model = TargetLM(model_name=args.target_model_name, device=args.target_model_cuda_id)  # Target LM model
    scanner = PromptInjectionScanner(model_name=args.assist_model_name, device=args.assist_model_cuda_id)  # Scanner model
    moderator = Moderator(model_name=args.moderation_model_name, device=args.moderation_model_cuda_id)  # Moderation model
    zero_shot_model = ZeroShot(model_name=args.zero_shot_model_name, device=args.zero_shot_model_cuda_id)  # Zero-shot model

    # Configure and load defense (Backtranslation-based defense mechanism)
    config = BacktranslationConfig()  # Configuration for backtranslation defense
    defense = load_defense(config, preloaded_model=target_model)  # Load the defense mechanism with the preloaded model
    defended_target_model = DefendedTargetLM(target_model, defense)  # Defended target LM model

    # Return a dictionary containing all initialized components
    return {
        'df': df,  # Loaded dataset
        'saver': saver,  # Saver for results
        'columns': columns,  # Columns to be saved
        'target_model': target_model,  # Target LM model
        'defended_target_model': defended_target_model,  # Defended LM model
        'scanner': scanner,  # Prompt Injection Scanner
        'moderator': moderator,  # Moderation model
        'zero_shot_model': zero_shot_model,  # Zero-shot evaluation model
        'substring_scanner': substring_scanner,  # Substring scanner
        'word_scanner': word_scanner,  # Word scanner
        'batch_size': batch_size  # Batch size for processing prompts
    }
