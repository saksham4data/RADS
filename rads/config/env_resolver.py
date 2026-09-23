import os

_TRUE = {'1', 'true', 'yes'}
_FALSE = {'0', 'false', 'no'}


def _as_bool(value):
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(f"invalid boolean: {value}")


def resolve_config(config_loader):
    applied = {}
    config = config_loader.config

    if 'RADS_SOURCE_URI' in os.environ:
        value = os.environ['RADS_SOURCE_URI']
        config.setdefault('runtime', {})['source_uri'] = value
        applied['runtime.source_uri'] = value

    if 'RADS_DEVICE' in os.environ:
        value = os.environ['RADS_DEVICE']
        config.setdefault('device', {})['compute'] = value
        applied['device.compute'] = value

    if 'RADS_MODEL_PATH' in os.environ:
        value = os.environ['RADS_MODEL_PATH']
        config.setdefault('device', {})['model_path'] = value
        applied['device.model_path'] = value

    if 'RADS_API_PORT' in os.environ:
        value = int(os.environ['RADS_API_PORT'])
        config.setdefault('api', {})['port'] = value
        applied['api.port'] = value

    if 'RADS_API_ENABLED' in os.environ:
        value = _as_bool(os.environ['RADS_API_ENABLED'])
        config.setdefault('api', {})['enabled'] = value
        applied['api.enabled'] = value

    return applied
