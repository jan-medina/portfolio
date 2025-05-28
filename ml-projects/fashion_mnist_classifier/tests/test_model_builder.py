from model.builder import CNNClassifierBuilderV2

def test_model_builder_variants():
    model1 = CNNClassifierBuilderV2(use_dropout=True, use_batchnorm=False)\
        .add_conv_layers().add_fc_layers().get_model()
    model2 = CNNClassifierBuilderV2(use_dropout=False, use_batchnorm=True)\
        .add_conv_layers().add_fc_layers().get_model()
    
    assert model1 is not None
    assert model2 is not None
    assert len(list(model1.children())) != len(list(model2.children()))
