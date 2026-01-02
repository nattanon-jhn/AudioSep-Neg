def get_model_class(model_type: str):
    if model_type == "ResUNet30":
        from models.resunet import ResUNet30
        return ResUNet30
    if model_type == "ResUNet30s":
        from models.resunet import ResUNet30s
        return ResUNet30s
    raise NotImplementedError(f"Unknown model_type: {model_type}")
