from matching.utils import supress_stdout, add_to_path, get_default_device
available_models = [
    "DesModel",
]


@supress_stdout
def get_matcher(
    retrieve_name="desmodel", device="cpu", checkpoint=None, *args, **kwargs
):
    if retrieve_name == "desmodel":
        from retrieve.Desmodel import DesModel
        return DesModel(device, checkpoint=checkpoint, *args, **kwargs)