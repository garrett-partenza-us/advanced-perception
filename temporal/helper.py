import torch
import numpy as np
from torchvision.transforms import Compose, Resize, InterpolationMode, Normalize, Lambda
import torchvision
from src.datasources import *

def make_transforms_JIF(
    input_size=(160, 160),
    output_size=(1054, 1054),
    interpolation=InterpolationMode.BICUBIC,
    normalize_lr=True,
    scene_classification_to_color=False,
    radiometry_depth=12,
    lr_bands_to_use="all",
    **kws,
):
    """ Make the transforms for the JIF dataset.
    The transforms normalize and resize the images to the appropriate sizes.

    Parameters
    ----------
    input_size : tuple
        The size of the (LR) input image, by default (160, 160).
    output_size : tuple
        The size of the (HR) output image, by default (1054, 1054).
    interpolation : torchvision.transforms.InterpolationMode, optional
        InterpolationMode to use when resizing the images, by default InterpolationMode.BILINEAR.
    normalize_lr : bool, optional
        A flag to normalize the LR images, by default True.
    scene_classification_to_color : bool, optional
        Converts the scene classification layer values to colors, by default False.
        See: https://www.sentinel-hub.com/faq/how-get-s2a-scene-classification-sentinel-2/
    **kws : dict
        The keyword arguments dictionary from which the input_size and output_size are fetched.

    Returns
    -------
    dict of {str : Callable}
        The LR, HR and scene classification transforms.
    """

    if radiometry_depth == 8:
        maximum_expected_hr_value = SPOT_MAX_EXPECTED_VALUE_8_BIT
    else:
        maximum_expected_hr_value = SPOT_MAX_EXPECTED_VALUE_12_BIT

    transforms = {}
    if lr_bands_to_use == "true_color":
        lr_bands_to_use = np.array(S2_ALL_12BANDS["true_color"]) - 1
    else:
        lr_bands_to_use = np.array(S2_ALL_BANDS) - 1

    if normalize_lr:
        normalize = Normalize(
            mean=JIF_S2_MEAN[lr_bands_to_use], std=JIF_S2_STD[lr_bands_to_use]
        )
    else:
        normalize = Compose([])

    transforms["lr"] = Compose(
        [
            Lambda(lambda lr_revisit: torch.as_tensor(lr_revisit)),
            normalize,
            Resize(size=input_size, interpolation=interpolation, antialias=True),
        ]
    )

    transforms["lrc"] = Compose(
        [
            Lambda(
                lambda lr_scene_classification: torch.as_tensor(lr_scene_classification)
            ),
            # Categorical
            Resize(size=input_size, interpolation=InterpolationMode.NEAREST),
            # Categorical to RGB; NOTE: interferes with FilterData
            SceneClassificationToColorTransform
            if scene_classification_to_color
            else Compose([]),
        ]
    )

    transforms["hr"] = Compose(
        [
            Lambda(
                lambda hr_revisit: torch.as_tensor(hr_revisit.astype(np.int32))
                / maximum_expected_hr_value
            ),
            Resize(size=output_size, interpolation=interpolation, antialias=True),
            Lambda(lambda high_res_revisit: high_res_revisit.clamp(min=0, max=1)),
        ]
    )

    transforms["hr_pan"] = Compose(
        [
            Lambda(
                lambda hr_panchromatic: torch.as_tensor(
                    hr_panchromatic.astype(np.int32)
                )
                / maximum_expected_hr_value
            ),  # sensor-dependent
            Resize(size=output_size, interpolation=interpolation, antialias=True),
            Lambda(
                lambda high_res_panchromatic: high_res_panchromatic.clamp(min=0, max=1)
            ),
        ]
    )
    # transforms["metadata"] = Compose([])
    return transforms