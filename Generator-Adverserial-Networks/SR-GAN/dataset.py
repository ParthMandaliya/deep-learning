from pathlib import Path
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image


high_res_width = 96
high_res_height = 96

low_res_width = high_res_width // 4
low_res_height = high_res_height // 4

highres_transform = T.Compose(
    [
        T.Resize((high_res_width, high_res_height),
                 interpolation=Image.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

lowres_transform = T.Compose(
    [
        T.Resize((low_res_width, low_res_height), interpolation=Image.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ]
)


class RealSRV3(Dataset):
    def __init__(self, root_dir: str):
        super(RealSRV3, self).__init__()
        self.root_dir = Path(root_dir)

        self.class_names = []
        for directory in self.root_dir.iterdir():
            for sub_directory in Path(directory).iterdir():
                for sub_sub_dir in Path(sub_directory).iterdir():
                    self.class_names.append(
                        str(Path(sub_sub_dir).joinpath(
                            self.root_dir, sub_directory, sub_sub_dir))
                    )

        self.data = []
        for image_filename in self.class_names:
            self.data += [x for x in Path(image_filename).iterdir()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(self.data[index])
        low_res = lowres_transform(image)
        high_res = highres_transform(image)

        return low_res, high_res
