def get_qwen_target_layers(model, n_layer, option="res2", every=1):
    map_names = dict(
        norm1=".input_layernorm",
        norm2=".post_attention_layernorm",
        res2="",
    )
    suffix = map_names[option]
    names = [name for name, _ in model.named_modules()]

    target_layers = {
        i: f"model.layers.{i}{suffix}" for i in range(0, n_layer, every)
    }
    if option == "norm1" or option == "norm2":
        target_layers[n_layer] = "model.norm"

    for target_layer in target_layers.values():
        assert target_layer in names, (target_layer, names)

    return target_layers


def get_embeddinggemma_target_layers(model, n_layer, option="res2", every=1):
    map_names = dict(
        norm1=".input_layernorm",
        norm2=".post_attention_layernorm",
        res2="",
    )
    suffix = map_names[option]
    names = [name for name, _ in model.named_modules()]

    target_layers = {
        i: f"model.layers.{i}{suffix}" for i in range(0, n_layer, every)
    }
    if option == "norm1" or option == "norm2":
        target_layers[n_layer] = "model.norm"

    for target_layer in target_layers.values():
        assert target_layer in names, (target_layer, names)

    return target_layers
