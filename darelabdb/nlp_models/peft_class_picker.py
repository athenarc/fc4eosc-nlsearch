from peft import LoraConfig

SUPPORTED_PEFT_CLASSES = {
    LoraConfig.__name__: LoraConfig,
}


def peft_class_picker(peft_name: str):
    """
    Returns the class of peft config on its name.
    Args:
        peft_name (str): The name of the requested peft class.
    """
    if peft_name not in SUPPORTED_PEFT_CLASSES:
        raise NotImplementedError(
            f"The class {peft_name} is not support. The supported classes "
            f"are: {list(SUPPORTED_PEFT_CLASSES.keys())}"
        )

    return SUPPORTED_PEFT_CLASSES[peft_name]
