import torchvision.transforms as transforms


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform_aug0 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

train_transform_center_crop = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

train_transform_aug01 = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


# def get_color_distortion(s=1.0):
#     # s is the strength of color distortion.
#     color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
#     rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
#     rnd_gray = transforms.RandomGrayscale(p=0.2)
#     color_distort = transforms.Compose([
#         rnd_color_jitter,
#         rnd_gray])
#     return color_distort


# color_distort = get_color_distortion()

# train_transform_aug1 = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomAutocontrast(0.5),
#     transforms.ToTensor(),
#     normalize
# ])

# train_transform_aug2 = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     color_distort,
#     transforms.ToTensor(),
#     normalize
# ])

# train_transform_aug3 = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     color_distort,
#     transforms.RandomAutocontrast(0.5),
#     transforms.ToTensor(),
#     normalize
# ])

# train_transform_aug4 = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomAutocontrast(0.5),
#     color_distort,
#     transforms.ToTensor(),
#     normalize
# ])


class ResizeImage(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            output size will be (size, size)
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

