import os


def read_data(root: str) -> list[list[str]]:
    """
    Read image paths from dataset root.

    Args:
        root (str): dataset root path.

    Returns:
        list[list[str]]: visible paths and infrared paths.
    """
    
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_root = root

    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF'] 

    train_visible_root = os.path.join(train_root, "vi")
    train_infrared_root= os.path.join(train_root, "ir")

    train_visible_path = [os.path.join(train_visible_root, i) for i in os.listdir(train_visible_root)
                  if os.path.splitext(i)[-1] in supported]
    train_infrared_path = [os.path.join(train_infrared_root, i) for i in os.listdir(train_infrared_root)
                  if os.path.splitext(i)[-1] in supported]

    train_visible_path.sort()
    train_infrared_path.sort()

    assert len(train_visible_path) == len(train_infrared_path), ' The length of vi and ir images does not match. vi: {}, ir: {}'.\
                                         format(len(train_visible_path), len(train_infrared_path))
    
    # print("Visible and Infrared images check finish")
    # print("{} visible images for training.".format(len(train_visible_path)))
    print("{} image pairs for training.".format(len(train_infrared_path)))

    train_low_light_path_list = [train_visible_path, train_infrared_path]
    return train_low_light_path_list