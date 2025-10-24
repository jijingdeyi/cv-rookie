import os.path
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import cv2
import torch.nn as nn
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pickle
from tqdm import tqdm
import threading
import random
from torchvision.transforms import functional as F

"""
This dataset from SAGE is used to train the teacher model. We have made some modifications to the original dataset.
1. We have changed the segmentation_aware_random_crop function order to ensure the label mask is cropped after the other images.
2. add the checkpoint resume capability to the mask generation process.
3. consistent add image masks every batch to prevent memory buildup.
4. remove the redundant manual seed setting.
5. use the use_mask_num to limit the number of masks processed, the original code has a logic error.
"""
class Data(Dataset):
    def __init__(self, mode, use_mask_num=20, cache_mask_num=50, crop_size=(600, 800), cache_dir=None, root_dir='/data/ykx/FMB'):
        self.root_dir = root_dir
        self.crop_size = crop_size

        # fetch the image list and extension information
        self.img_list = []
        self.extensions = {}

        for filename in os.listdir(os.path.join(self.root_dir, 'Vis')):
            name, ext = os.path.splitext(filename)
            self.img_list.append(name)
            self.extensions[name] = ext

        self.img_dir = root_dir

        # confirm the number of infrared images is equal to the number of visible light images
        assert len(os.listdir(os.path.join(self.img_dir, 'Ir'))
                   ) == len(self.img_list)

        assert mode == 'train' or mode == 'test', "dataset mode not specified"
        self.mode = mode
        if mode == 'train':
            # do not use RandomResizedCrop, we will customize the crop logic
            self.transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(p=0.5)])
        elif mode == 'test':
            self.transform = transforms.Compose([])

        # the number of masks generated per image in the cache
        self.cache_mask_num = cache_mask_num
        # the number of masks actually used, cannot exceed the number in the cache
        self.use_mask_num = min(use_mask_num, cache_mask_num)
        self.totensor = transforms.ToTensor()

        # set the cache directory
        self.cache_dir = cache_dir or os.path.join(self.root_dir, 'Mask_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        self.mask_cache = {}

        # check if there is a cache file - note that here we use cache_mask_num as part of the cache file name
        cache_file = os.path.join(
            self.cache_dir, f'mask_cache_{mode}_{cache_mask_num}.pkl')
        if os.path.exists(cache_file):
            print(f"Loading mask cache from {cache_file}")
            self.mask_cache = self._load_cache_file(cache_file)
            print(
                f"Loaded masks for {len(self.mask_cache)} images (cached: {cache_mask_num}, using: {use_mask_num})")
        else:
            # initialize the SAM model and generate all masks
            print(
                f"Initializing SAM model and generating {cache_mask_num} masks per image...")
            self._initialize_sam_and_generate_masks(cache_file)

        # for tracking whether the zero mask warning has been printed
        self.zero_mask_warning_printed = False

    def _initialize_sam_and_generate_masks(self, cache_file):
        # initialize the SAM model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sam = sam_model_registry["vit_b"](
            checkpoint='/data/ykx/sam_vit_b_01ec64.pth').to(device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=128,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
            output_mode='binary_mask',
        )

        # check for existing progress and resume from checkpoint
        progress_file = cache_file.replace('.pkl', '_progress.pkl')
        start_idx = 0

        if os.path.exists(progress_file):
            print(f"Found existing progress file: {progress_file}")
            with open(progress_file, 'rb') as f:
                progress_data = pickle.load(f)
                start_idx = progress_data.get('last_processed_idx', 0)
                print(
                    f"Resuming from image {start_idx + 1}/{len(self.img_list)}")
        else:
            print("Starting mask generation from the beginning")

        # generate masks with checkpoint resume capability
        for idx in tqdm(range(start_idx, len(self.img_list)), desc="Generating masks"):
            name_0 = self.img_list[idx]
            # get the extension, default is .png
            ext = self.extensions.get(name_0, '.png')

            ir_path_0 = os.path.join(self.img_dir, 'Ir', name_0 + ext)
            vis_path_0 = os.path.join(self.img_dir, 'Vis', name_0 + ext)

            # read the images
            ir_img = cv2.imread(ir_path_0)
            vis_img = cv2.imread(vis_path_0)

            # generate masks
            ir_patches = mask_generator.generate(ir_img)
            ir_patches.sort(key=lambda x: x['area'], reverse=True)

            vis_patches = mask_generator.generate(vis_img)
            vis_patches.sort(key=lambda x: x['area'], reverse=True)

            # store masks - use cache_mask_num
            ir_masks = []
            vis_masks = []

            for i in range(min(self.cache_mask_num, len(ir_patches), len(vis_patches))):
                ir_masks.append(ir_patches[i]['segmentation'])
                vis_masks.append(vis_patches[i]['segmentation'])

            # store masks in batch_data temporarily (not in self.mask_cache)
            if not hasattr(self, 'batch_data'):
                self.batch_data = {}

            self.batch_data[name_0] = {
                'ir_masks': ir_masks,
                'vis_masks': vis_masks
            }

            # incremental save every 50 samples to prevent memory buildup
            if (idx + 1) % 50 == 0:
                # save current batch incrementally
                with open(cache_file, 'ab') as f:  # 'ab' = append binary mode
                    pickle.dump(self.batch_data, f)

                # clear batch_data from memory to prevent buildup
                self.batch_data = {}

                # save progress
                progress_data = {'last_processed_idx': idx}
                with open(progress_file, 'wb') as f:
                    pickle.dump(progress_data, f)
                print(f"Batch saved at image {idx + 1}/{len(self.img_list)}")

        # save any remaining masks in batch_data
        if hasattr(self, 'batch_data') and self.batch_data:
            with open(cache_file, 'ab') as f:
                pickle.dump(self.batch_data, f)
            print(f"Saved final batch with {len(self.batch_data)} images")

        # consolidate incremental cache files into final cache
        print("Consolidating incremental cache files...")
        self._consolidate_incremental_cache(cache_file)

        # remove progress file when complete
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print("Progress file cleaned up")

        print(
            f"Mask generation complete. Saved {self.cache_mask_num} masks per image to {cache_file}")

    def _consolidate_incremental_cache(self, cache_file):
        """Consolidate incremental cache files into a single cache file"""
        consolidated_cache = {}

        # read all incremental data
        with open(cache_file, 'rb') as f:
            while True:
                try:
                    single_image_data = pickle.load(f)
                    consolidated_cache.update(single_image_data)
                except EOFError:
                    break

        # save consolidated cache
        with open(cache_file, 'wb') as f:
            pickle.dump(consolidated_cache, f)

        print(f"Consolidated cache with {len(consolidated_cache)} images")

    def _load_cache_file(self, cache_file):
        """Load cache file, handling both consolidated and incremental formats"""
        try:
            # first try to load as consolidated cache
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            return cache
        except:
            # if that fails, try to load as incremental cache
            consolidated_cache = {}
            with open(cache_file, 'rb') as f:
                while True:
                    try:
                        single_image_data = pickle.load(f)
                        consolidated_cache.update(single_image_data)
                    except EOFError:
                        break
            return consolidated_cache

    def random_crop(self, image, seed=2026, target_size=(600, 800)):
        """
        simple random crop function, does not depend on masks
        """
        # set random seed to ensure consistency
        torch.manual_seed(seed)
        random.seed(seed)

        c, h, w = image.shape
        target_h, target_w = target_size

        # random crop the entire image
        if h <= target_h:
            i = 0
            crop_h = h
        else:
            i = torch.randint(0, h - target_h + 1, (1,)).item()
            crop_h = target_h

        if w <= target_w:
            j = 0
            crop_w = w
        else:
            j = torch.randint(0, w - target_w + 1, (1,)).item()
            crop_w = target_w

        cropped = image[:, i:i+crop_h, j:j+crop_w]

        # if the size of the cropped image does not match the target size, resize it
        if cropped.shape[1] != target_h or cropped.shape[2] != target_w:
            cropped = F.resize(cropped, target_size)

        return cropped

    def segmentation_aware_random_crop(self, image, mask, seed, target_size):
        """
        random crop within the bounding box containing the segmentation region, then resize
        Handle the case where mask is all zeros

        Args:
            image: input image tensor [C, H, W]
            mask: segmentation mask tensor [H, W] or [1, H, W]
            seed: random seed
            target_size: target size (h, w)

        Returns:
            cropped and resized image
        """
        # set random seed to ensure consistency
        torch.manual_seed(seed)
        random.seed(seed)

        # ensure the mask is 2D
        if mask.dim() == 3 and mask.size(0) == 1:
            mask = mask.squeeze(0)

        # get the size of the image and mask
        c, h, w = image.shape
        target_h, target_w = target_size

        # find the coordinates of the non-zero regions in the mask
        non_zero_indices = torch.nonzero(mask > 0.5, as_tuple=False)

        # if the mask is empty (all zero), perform ordinary random crop
        if len(non_zero_indices) == 0:
            # print the warning only once
            if not self.zero_mask_warning_printed:
                print("Warning: Some masks are zero, performing standard random crop")
                self.zero_mask_warning_printed = True

            # use simple random crop
            return self.random_crop(image, seed, target_size)
        else:
            # get the bounding box of the mask
            min_y, min_x = map(int, non_zero_indices.min(0)[0])
            max_y, max_x = map(int, non_zero_indices.max(0)[0])

            # calculate the size of the bounding box
            box_h = max_y - min_y + 1
            box_w = max_x - min_x + 1

            # Ensure bounding box is at least as large as target size
            # If bounding box is smaller than target size, expand the bounding box
            if box_h < target_h:
                padding = target_h - box_h
                min_y = max(0, min_y - padding // 2)
                max_y = min(h - 1, max_y + padding // 2 + padding % 2)
                box_h = max_y - min_y + 1

            if box_w < target_w:
                padding = target_w - box_w
                min_x = max(0, min_x - padding // 2)
                max_x = min(w - 1, max_x + padding // 2 + padding % 2)
                box_w = max_x - min_x + 1

            # Randomly select crop starting point within bounding box
            if box_h > target_h:
                i = min_y + torch.randint(0, box_h - target_h + 1, (1,)).item()
            else:
                i = min_y

            if box_w > target_w:
                j = min_x + torch.randint(0, box_w - target_w + 1, (1,)).item()
            else:
                j = min_x

            # Ensure crop region does not exceed image boundaries
            i = min(max(0, i), h - target_h)
            j = min(max(0, j), w - target_w)

            # Perform cropping
            crop_h = min(h - i, target_h)
            crop_w = min(w - j, target_w)
            cropped = image[:, i:i+crop_h, j:j+crop_w]

        # If cropped size does not match target size, resize
        if cropped.shape[1] != target_h or cropped.shape[2] != target_w:
            cropped = F.resize(cropped, target_size)

        return cropped

    def __getitem__(self, idx):
        seed = idx + 2026

        name_0 = self.img_list[idx]
        # Get extension, default to .png
        ext = self.extensions.get(name_0, '.png')

        label = []

        label_item_path = os.path.join(self.img_dir, 'Label', name_0 + ext)
        label_mask = cv2.imread(label_item_path)
        label_mask_tensor = self.totensor(
            cv2.cvtColor(label_mask, cv2.COLOR_BGR2GRAY))

        ir_path_0 = os.path.join(self.img_dir, 'Ir', name_0 + ext)
        vis_path_0 = os.path.join(self.img_dir, 'Vis', name_0 + ext)
        ir_0 = cv2.imread(ir_path_0)
        vi_0 = cv2.imread(vis_path_0)
        ir_0_tensor = self.totensor(cv2.cvtColor(ir_0, cv2.COLOR_BGR2GRAY))
        vi_0_tensor = self.totensor(
            cv2.cvtColor(vi_0, cv2.COLOR_BGR2YCrCb))  # CHW

        # use original label mask to guide other images cropping
        if self.mode == 'train':
            # Check if label mask is all zeros
            if torch.sum(label_mask_tensor) > 0:
                ir_0_tensor = self.segmentation_aware_random_crop(
                    ir_0_tensor, label_mask_tensor, seed, self.crop_size)
                vi_0_tensor = self.segmentation_aware_random_crop(
                    vi_0_tensor, label_mask_tensor, seed, self.crop_size)

                # Finally crop the label mask to maintain consistency
                label_mask_tensor = self.segmentation_aware_random_crop(
                    label_mask_tensor, label_mask_tensor, seed, self.crop_size)
            else:
                # If label mask is all zeros, use simple random cropping
                if not self.zero_mask_warning_printed:
                    print(
                        "Warning: Label mask is zero, performing standard random crop")
                    self.zero_mask_warning_printed = True

                # Use same seed for simple random cropping
                label_mask_tensor = self.random_crop(
                    label_mask_tensor, seed, self.crop_size)
                ir_0_tensor = self.random_crop(
                    ir_0_tensor, seed, self.crop_size)
                vi_0_tensor = self.random_crop(
                    vi_0_tensor, seed, self.crop_size)

            # Apply other transformations (such as horizontal flip)
            torch.manual_seed(seed)
            label_mask_tensor = self.transform(label_mask_tensor)
            ir_0_tensor = self.transform(ir_0_tensor)
            vi_0_tensor = self.transform(vi_0_tensor)

        y_0 = vi_0_tensor[0, :, :].unsqueeze(dim=0).clone()
        cb = vi_0_tensor[1, :, :].unsqueeze(dim=0)
        cr = vi_0_tensor[2, :, :].unsqueeze(dim=0)

        irs = []
        ys = []

        # Get masks from cache
        cached_masks = self.mask_cache.get(name_0)
        if cached_masks:
            ir_img = cv2.imread(ir_path_0)
            vis_img = cv2.imread(vis_path_0)

            # Count valid masks
            valid_mask_count = 0

            # Use use_mask_num to limit the number of masks processed
            for i in range(min(self.cache_mask_num, len(cached_masks['ir_masks']))):
                # Check if mask is all zeros
                ir_mask = cached_masks['ir_masks'][i]
                vis_mask = cached_masks['vis_masks'][i]

                if not np.any(ir_mask) or not np.any(vis_mask):
                    # Skip all-zero masks without printing warnings
                    continue

                # Apply infrared mask
                ir_position = ~ir_mask
                ir_masked = ir_img.copy()
                ir_masked[ir_position] = 0

                # Apply visible light mask
                vis_position = ~vis_mask
                vis_masked = vis_img.copy()
                vis_masked[vis_position] = 0

                try:
                    ir_2_tensor = self.totensor(
                        cv2.cvtColor(ir_masked, cv2.COLOR_BGR2GRAY))
                    vi_2_tensor = self.totensor(
                        cv2.cvtColor(vis_masked, cv2.COLOR_BGR2YCrCb))

                    # In training mode, use segmentation-aware random cropping
                    if self.mode == 'train':
                        # Use same cropping and transformation as original images
                        if torch.sum(label_mask_tensor) > 0:
                            ir_2_tensor = self.segmentation_aware_random_crop(
                                ir_2_tensor, label_mask_tensor, seed, self.crop_size)
                            vi_2_tensor = self.segmentation_aware_random_crop(
                                vi_2_tensor, label_mask_tensor, seed, self.crop_size)
                        else:
                            ir_2_tensor = self.random_crop(
                                ir_2_tensor, seed, self.crop_size)
                            vi_2_tensor = self.random_crop(
                                vi_2_tensor, seed, self.crop_size)

                        torch.manual_seed(seed)
                        ir_2_tensor = self.transform(ir_2_tensor)
                        vi_2_tensor = self.transform(vi_2_tensor)

                    y = vi_2_tensor[0, :, :].unsqueeze(dim=0)

                    irs.append(ir_2_tensor)
                    ys.append(y)

                    # Increment valid mask count
                    valid_mask_count += 1

                    # If enough valid masks collected, exit loop
                    if valid_mask_count >= self.use_mask_num:
                        break

                except Exception as e:
                    # Continue on error without printing detailed error info
                    continue

        # If insufficient masks, fill with original images
        while len(irs) < self.use_mask_num:
            irs.append(ir_0_tensor.clone())
            ys.append(y_0.clone())

        ys_0 = torch.cat(ys, dim=0)
        irs_0 = torch.cat(irs, dim=0)

        result = {'name': name_0, 'irs': irs_0, 'ys': ys_0, 'label': label,
                  'ir': ir_0_tensor, 'y': y_0, 'cb': cb, 'cr': cr, 'label_mask': label_mask_tensor}

        return result

    def trans(self, x, seed):
        torch.manual_seed(seed)
        x = self.transform(x)
        return x

    def __len__(self):
        return len(self.img_list)
