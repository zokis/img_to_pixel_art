import argparse
import numpy as np
import multiprocessing
from PIL import Image
from scipy.spatial import distance


def split_into_blocks(image_array, block_size: int) -> list:
    height, width = image_array.shape[:2]
    blocks = []
    for i in range(0, height - block_size + 1, block_size):
        for j in range(0, width - block_size + 1, block_size):
            blocks.append(
                image_array[i : i + block_size, j : j + block_size].reshape(-1, 3)
            )  # Flatten the block
    return blocks


def common_divisors(n: int, min_block_size: int, max_block_size: int) -> set:
    divisors = set()
    for i in range(min_block_size, max_block_size + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    return divisors


def find_best_block_size(
    width: int, height: int, min_block_size: int = 4, max_block_size: int = 64
) -> list:
    """
    Finds common divisors for image dimensions and suggests the best block sizes.
    :param width: Image width.
    :param height: Image height.
    :param min_block_size: Minimum block size considered useful.
    :param max_block_size: Maximum block size considered useful.
    :return: A list of suggested block sizes.
    """
    width_divisors = set(common_divisors(width, min_block_size, max_block_size))
    height_divisors = set(common_divisors(height, min_block_size, max_block_size))
    common_block_sizes = sorted(width_divisors.intersection(height_divisors))
    return [size for size in common_block_sizes if size <= max_block_size]


def calculate_color_homogeneity(blocks: list) -> float:
    """
    Calculates the color homogeneity of the blocks,
    returning the average color distance for each block.
    :param blocks: List of pixel blocks.
    :return: List of average color distances for each block.
    """
    distances = []
    for block in blocks:
        block_mean_color = np.mean(block, axis=0)
        distances.append(np.mean(np.linalg.norm(block - block_mean_color, axis=1)))
    return np.mean(distances)


def calculate_color_homogeneity_parallel(block_size, image_array):
    blocks = split_into_blocks(image_array, block_size)
    average_distance = calculate_color_homogeneity(blocks)
    return block_size, average_distance


def find_best_block_size_by_color_with_threshold(
    image_array, possible_block_sizes: list, threshold: float = 1.0
) -> int:
    """
    Finds the best block size based on color homogeneity,
    preferring larger blocks if the difference in homogeneity is small.
    :param image_array: Numpy array of the image.
    :param possible_block_sizes: List of possible block sizes.
    :param threshold: The maximum allowed difference in
    homogeneity to prefer a larger block size.
    :return: The chosen block size.
    """
    print("Evaluating best block size based on color homogeneity")
    best_block_size = 4
    lowest_distance = float("inf")

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(
            calculate_color_homogeneity_parallel,
            [(block_size, image_array) for block_size in possible_block_sizes],
        )

    for block_size, average_distance in sorted(results):
        print(
            f"  Average color distance for block size {block_size}: {average_distance}"
        )
        if average_distance < lowest_distance + threshold:
            lowest_distance = average_distance
            best_block_size = block_size

    print(f"Best block size found (with multiprocessing): {best_block_size}")
    return best_block_size


def find_representative_color(block: tuple) -> tuple:
    """
    Finds the representative color of the block,
    which is the color with the least total distance
    to all other colors in the block.
    :param block: List of RGB color tuples from a block.
    :return: The representative color (RGB tuple).
    """
    distances = distance.cdist(block, block, "euclidean")
    np.fill_diagonal(distances, 0)
    total_distances = np.sum(distances, axis=1)
    return block[np.argmin(total_distances)]


def create_reduced_image(image_array, block_size: int):
    """
    Creates a new image with the representative color for each block.
    :param image_array: Numpy array of the original image.
    :param block_size: Size of the block to be used.
    :return: PIL Image object of the reduced image.
    """
    print(f"Creating reduced image with block size {block_size}")
    height, width, _ = image_array.shape
    reduced_height = height // block_size
    reduced_width = width // block_size
    reduced_image_array = np.zeros((reduced_height, reduced_width, 3), dtype=np.uint8)

    blocks = split_into_blocks(image_array, block_size)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        representative_colors = pool.map(find_representative_color, blocks)

    for i, color in enumerate(representative_colors):
        block_x = i % reduced_width
        block_y = i // reduced_width
        reduced_image_array[block_y, block_x] = color[:3]

    return Image.fromarray(reduced_image_array)


def main(image_path, n_colors=256, specified_block_size=None):
    print(f"Processing image: {image_path}")
    # Load the image
    image = (
        Image.open(image_path)
        .quantize(colors=n_colors, method=Image.MEDIANCUT)
        .convert("RGB")
    )
    image_array = np.array(image)
    height, width = image_array.shape[:2]

    if specified_block_size is None:
        # Find the best block sizes
        best_block_sizes = find_best_block_size(
            width, height, min_block_size=4, max_block_size=64
        )
        print(f"Best block sizes: {best_block_sizes}")

        # Find the best block size by color
        best_block_size_by_color = find_best_block_size_by_color_with_threshold(
            image_array, best_block_sizes
        )
        print(f"Best block size by color: {best_block_size_by_color}")
    else:
        best_block_size_by_color = specified_block_size

    # Create a new image with the representative color for each block
    reduced_image = create_reduced_image(image_array, best_block_size_by_color)

    pixeled_image_path = (
        image_path.rsplit(".", 1)[0] + f"_{best_block_size_by_color}_pixeled.png"
    )

    reduced_image.save(pixeled_image_path, format="PNG", optimize=True)

    print(f"New image saved as: {pixeled_image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to pixel art.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Block size for pixelation (optional)",
    )
    parser.add_argument(
        "--n_colors",
        type=int,
        default=256,
        help="Number of colors to reduce to (optional, default=256)",
    )

    args = parser.parse_args()

    main(args.image_path, n_colors=args.n_colors, specified_block_size=args.block_size)
