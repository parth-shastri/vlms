import torch
import numpy as np
from typing import List, Tuple, Union, Iterable
from PIL import Image

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    # This is just the prefix prompt, the whole completion will be the completion.
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image


def resize(
    image: Image.Image,
    size: Tuple[int, int],
    resample: Image.Resampling,
    reducing_gap: float | None = None,
):
    image = image.resize(size=size, resample=resample, reducing_gap=reducing_gap)
    return image


def process_images(
    images: List[Image.Image],
    size: Tuple[int, int],
    resample: Image.Resampling,
    rescale_factor: float,
    image_mean: List[float],
    image_std: List[float],
) -> List[np.ndarray]:
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    # Convert each image into numpy array
    images = [np.array(image) for image in images]
    # Rescale the pixel values in the range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Normalize the images to have 0 mean & 1 std
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # Move the channel dimension to the first dimension
    images = [image.transpose(2, 0, 1) for image in images]
    return images


class PaliGemmaProcessor:

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_len = num_image_tokens
        self.image_size = image_size

        # Tokenizer described here
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # these are the normalized locations for bounding boxes
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # These tokens are used for segmentation (VQ-VAE quantized tokens (128))
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # we will add the BOS & EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = False,
        return_token_type_ids=True,
        **kwargs,
    ) -> dict:
        assert (
            len(images) == 1 and len(text) == 1
        ), f"PaliGemmaProcessor currently supports single image & text, {len(images)} images for {len(text)} prompts"

        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )
        # convert the list of numpy arrays to a single numpy array of size (B_S, C, H, W)
        pixel_values = np.stack(pixel_values, axis=0)
        # convert the numpy array to Pytorch tensor
        pixel_values = torch.tensor(pixel_values)

        # The 'text' input is the prompt or the prefix prompt
        suffix = kwargs.pop("suffix", None)
        return_token_type_ids = (
            True if suffix is not None else return_token_type_ids
        )  # Always true if the suffix is present

        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_len,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Returns the input_ids and attention_mask as Pytorch tensors
        inputs = self.tokenizer(
            input_strings,
            text_pair=suffix,
            return_token_type_ids=return_token_type_ids,
            return_tensors=kwargs.pop("return_tensors", "pt"),
            padding=padding,
            truncation=truncation,
        )

        if return_token_type_ids:
            labels = inputs["token_type_ids"].masked_fill(inputs["token_type_ids"] == 0, -100)
            inputs.update({"labels": labels})
        return_data = {"pixel_values": pixel_values, **inputs}
        return return_data
