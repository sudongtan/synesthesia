from fastai.text.models.transformer import tfmerXL_lm_config, Activation


def default_config():
    config = tfmerXL_lm_config.copy()
    config["act"] = Activation.GeLU

    config["mem_len"] = 512
    config["d_model"] = 512
    config["d_inner"] = 2048
    config["n_layers"] = 16

    config["n_heads"] = 8
    config["d_head"] = 64
    config["encode_position"] = True

    return config

