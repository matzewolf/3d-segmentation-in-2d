from torch.utils.data import DataLoader

from dataset import ShapeNetPartDataset


def visualize_2d(split: str = "val"):
    dataset = ShapeNetPartDataset(split=split)
    dataloader = DataLoader(dataset)
    for (points_3d_image, parts_image) in dataloader:
        print(f"{points_3d_image.min()=}")
        print(f"{points_3d_image.max()=}")
        print(f"{parts_image.min()=}")
        print(f"{parts_image.max()=}")


if __name__ == '__main__':
    visualize_2d()
