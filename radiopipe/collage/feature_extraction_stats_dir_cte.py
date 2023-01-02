import numpy as np
import SimpleITK as sitk
from scipy.stats import kurtosis, skew
from skimage.util import img_as_float
from utility import logger_init, now, path_in, path_out

from .collageradiomics import Collage

logger = logger_init(__file__.split("/")[-1])


path_images = path_in("Data/Crohns_Disease/CCF_data/Postprocessed/ISO/BFC")
path_labels = path_in("Data/Crohns_Desease/")

gen_images = path_images.glob("*.nii")
lst_images = list(gen_images)

gen_labels = path_labels.glob("*.nii")
lst_labels = list(gen_labels)

current_time = now()


path_output_features_stats = path_out(rf"Data/Crohns_Disease")


def compute_collage(image, mask, image_name, haralick_windows=[3, 5, 7, 9, 11]):
    windows_length = len(haralick_windows)

    feats = np.zeros((13 * windows_length * 2, 4), dtype=np.double)

    for window_idx, haralick_window_size in enumerate(haralick_windows):
        try:
            collage = Collage(
                image,
                mask,
                svd_radius=5,
                verbose_logging=True,
                num_unique_angles=64,
                haralick_window_size=haralick_window_size,
            )

            collage_feats = collage.execute()

            for orientation in range(2):
                for collage_idx in range(13):
                    k = window_idx * collage_idx * orientation
                    feat = collage_feats[:, :, :, collage_idx, orientation].flatten()
                    feat = feat[~np.isnan(feat)]

                    feats[k, 0] = feat.mean()
                    feats[k, 1] = feat.std()
                    feats[k, 2] = skew(feat)
                    feats[k, 3] = kurtosis(feat)

                    print(feats[k, :])

        except ValueError as err:
            print(f"VALUE ERROR- {err}")
            logger.error(f"VALUE ERROR- {err}")

        except Exception as err:
            print(f"EXCEPTION- {err}")
            logger.error(f"EXCEPTION- {err}")

    print("FEATS SHAPE", feats.shape)
    np.save(
        rf"{path_output_features_stats}/{image_name}.npy",
        feats,
    )


for path_image in lst_images:

    image_sitk = sitk.ReadImage(str(path_image))
    image_array = sitk.GetArrayFromImage(image_sitk)
    image = img_as_float(image_array)

    id_image = image.name.split("_")[0]
    label_name = id_image + "a.npy"
    path_label = path_in("Data/Florian_CTE_processed/labels_Res_Bin") / label_name

    image_name = image.stem

    mask = np.load(path_label)

    image = np.swapaxes(image, 0, 2)
    mask = np.swapaxes(mask, 0, 2)

    compute_collage(
        image,
        mask,
        image_name=image_name,
        haralick_windows=[3, 5, 7, 9, 11],
    )

    print(id_image)
