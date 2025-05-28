from utils.config import config

def test_config_fields_exist():
    assert hasattr(config, 'batch_size')
    assert config.batch_size > 0
