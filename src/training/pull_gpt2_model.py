from training.gpt_download import download_and_load_gpt2

openai_settings, openai_params = download_and_load_gpt2(
      model_size="124M"
    , models_dir="gpt2"
    )
