from .backtranslation import BackTranslationDefense, BacktranslationConfig

def args_to_defense_config(args) -> BacktranslationConfig:
    """
    Converts command-line arguments into a BacktranslationConfig instance.

    Parameters:
        args: Command-line arguments containing the defense method.

    Returns:
        BacktranslationConfig: Configured defense configuration.
    """
    config = BacktranslationConfig(args.defense_method)
    config.load_from_args(args)
    return config

def load_defense(defense_config: BacktranslationConfig, preloaded_model=None) -> BackTranslationDefense:
    """
    Loads the appropriate defense method based on the provided defense configuration.

    Parameters:
        defense_config (BacktranslationConfig): The defense configuration.
        preloaded_model: Optional preloaded model for the defense method.

    Returns:
        BackTranslationDefense: The defense instance configured according to the defense configuration.
    """
    defense_method = defense_config.defense_method

    if defense_method.startswith("backtranslation"):
        # Extract threshold value if present
        if 'threshold' in defense_method:
            try:
                threshold = float(defense_method.split('_')[-1])
                defense_config.backtranslation_threshold = -threshold if threshold > 0 else threshold
            except ValueError:
                defense_config.backtranslation_threshold = -2.0
        else:
            defense_config.backtranslation_threshold = -2.0

    return BackTranslationDefense(defense_config, preloaded_model)
