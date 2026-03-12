import cv2
from cv2.typing import MatLike
import matplotlib.pyplot as plt
import argparse
import os
from objectDetection import detect_objects


def resize_for_display(img: MatLike, display_width: int = 800) -> MatLike:
    h, w = img.shape[:2]
    if w <= 0:
        return img
    if w <= display_width:
        return img
    scale = display_width / w
    display_height = int(h * scale)
    return cv2.resize(img, (display_width, display_height))


def main():
    parser = argparse.ArgumentParser(description="Run DNN object detection on an image.")
    parser.add_argument("--image", default="test.jpg", help="Input image path (default: test.jpg)")
    parser.add_argument("--show", action="store_true", help="Display result image on screen (Matplotlib).")
    parser.add_argument("--out", help="Save result image to this path.")
    parser.add_argument("--display-width", type=int, default=800, help="Display width used with --show.")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"Cannot load image: {args.image}")
        return

    result = detect_objects(image)

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        ok = cv2.imwrite(args.out, result.image)
        if not ok:
            print(f"Failed to save image to: {args.out}")
            return
        print(f"Saved result to: {args.out}")

    if args.show:
        display_img = resize_for_display(result.image, display_width=args.display_width)
        result_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(result_rgb)
        plt.axis("off")
        plt.show()

    if (not args.show) and (not args.out):
        print("No output selected. Use --show to display and/or --out to save.")


if __name__ == "__main__":
    main()
