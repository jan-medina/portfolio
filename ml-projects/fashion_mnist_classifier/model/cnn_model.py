# model/cnn_model.py

from model.builder import CNNClassifierBuilder, CNNClassifierBuilderV2


def build_model(version='v1', **kwargs):
    """
    Construye un modelo CNN configurable.
    Parámetros:
        version (str): 'v1' para builder simple, 'v2' para configurable.
        kwargs: parámetros extra como use_dropout, use_batchnorm.
    """
    if version == 'v2':
        builder = CNNClassifierBuilderV2(**kwargs)
    else:
        builder = CNNClassifierBuilder()

    model = (
        builder
        .add_input_layer()
        .add_conv_layers()
        .add_fc_layers()
        .get_model()
    )
    return model
