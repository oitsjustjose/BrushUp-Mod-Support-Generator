from blendmodes.blend import blendLayers, BlendType
import os
import cv2
import numpy as np
from PIL import Image
import shutil


def get_plank_dominant_color(plank_path: str) -> np.ndarray:
    colors = []
    img: cv2.Mat = cv2.imread(plank_path)

    # super stupid method
    rows, cols, _ = img.shape
    color = img[cols // 2, rows // 2]
    colors.append(color)

    # averaging method
    avg_color_per_row = np.average(img, axis=0)  # type: ignore
    avg_color = np.average(avg_color_per_row, axis=0)
    colors.append(avg_color)

    pixels = np.float32(img.reshape(-1, 3))  # type: ignore
    n_colors = 16
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_PP_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    colors.append(palette[np.argmax(counts)])
    colors.append(palette[np.argmin(counts)])

    ret = np.concatenate((np.mean(colors, axis=0), [255]))
    return ret


def generate_textures(modid: str, plank: str) -> None:
    def to_pillow(img: cv2.Mat) -> Image.Image:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        img_pil = Image.fromarray(img, mode="RGBA")
        return img_pil

    plank_color = get_plank_dominant_color(f"./textures/{modid}/{plank}")

    # --DISABLED-- Attempt to dynamically choose a darker base texture
    # To re-enable, if change paths to {'light' if use_light else 'dark'}
    # use_light = np.average(plank_color[:3], axis=0) > 128
    for ref_text_path in os.listdir("./references/textures/dark"):
        ref_text = cv2.imread(
            f"./references/textures/dark/{ref_text_path}",
            cv2.IMREAD_UNCHANGED,
        )

        # Create bitmask, apply and fill with plank color
        alpha = np.sum(ref_text, axis=-1) > 0
        overlay = ref_text.copy()
        overlay[alpha > 0] = plank_color

        overlay = to_pillow(overlay)
        ref_text = to_pillow(ref_text)
        # Ugh, why does overlay work so well -_-
        output = blendLayers(ref_text, overlay, BlendType.OVERLAY)

        plank_nm = plank.replace("_planks", "").replace(".png", "")
        root = f"./out/assets/{modid}/textures/block"
        os.makedirs(root, exist_ok=True)

        output.save(f"{root}/{plank_nm}_{ref_text_path.replace('./', '')}")


def generate_data(modid: str, plank: str) -> None:
    plank_nm = plank.replace("_planks", "").replace(".png", "")
    # Skip over reading these as JSON, we can just do a dummy replace
    for dtype in ["blockstates", "models/block"]:
        for ref in os.listdir(f"./references/{dtype}"):
            root = f"./out/assets/{modid}/{dtype}"
            os.makedirs(root, exist_ok=True)

            with open(f"./references/{dtype}/{ref}", "r") as handle:
                data = handle.read()

            data = data.replace("<MODID>", modid).replace("<PLANK>", plank_nm)
            with open(f"./out/assets/{modid}/{dtype}/{plank_nm}{ref}", "w") as handle:
                handle.write(data)


def main() -> None:
    for modid in os.listdir("./textures"):
        if not os.path.isdir(f"./textures/{modid}"):
            print(f"[i] ./textures/{modid} is not a directory: skipping")
            continue

        for plank in os.listdir(f"./textures/{modid}"):
            generate_textures(modid, plank)
            generate_data(modid, plank)

    for modid in os.listdir("./overrides"):
        if not os.path.isdir(f"./overrides/{modid}"):
            print(f"[i] ./overrides/{modid} is not a directory: skipping")
            continue
        for texture in os.listdir(f"./overrides/{modid}"):
            shutil.copyfile(
                f"./overrides/{modid}/{texture}",
                f"./out/assets/{modid}/textures/block/{texture}",
            )


if __name__ == "__main__":
    main()
