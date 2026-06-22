def resolve_model(model: str) -> str:
    if "/" not in model:
        return f"noxus/{model}"
    return model


def provider_of(model: str) -> str:
    resolved = resolve_model(model)
    return resolved.split("/", 1)[0]


__all__ = [
    "resolve_model",
    "provider_of",
]
