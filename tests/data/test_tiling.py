# tests/data/test_tiling.py
import os
import tempfile
from pathlib import Path
from PIL import Image
import pytest
from src.data.tiling import tile_image, TileInfo


def make_test_image(path: str, size: tuple) -> None:
    img = Image.new("RGB", size, color=(100, 150, 200))
    img.save(path)


def test_tile_count_exact_fit():
    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "test.jpg")
        make_test_image(img_path, (1280, 1280))
        tiles = tile_image(img_path, os.path.join(tmp, "tiles"), tile_size=640)
        assert len(tiles) == 4  # 2 cols × 2 rows


def test_tile_count_partial_edge_excluded():
    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "test.jpg")
        make_test_image(img_path, (900, 900))  # 900 / 640 = 1.4, only 1 full tile per axis
        tiles = tile_image(img_path, os.path.join(tmp, "tiles"), tile_size=640)
        assert len(tiles) == 1


def test_tile_files_created():
    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "test.jpg")
        make_test_image(img_path, (1280, 1280))
        tiles = tile_image(img_path, os.path.join(tmp, "tiles"), tile_size=640)
        assert all(Path(t.tile_path).exists() for t in tiles)


def test_tile_info_fields():
    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "scene.jpg")
        make_test_image(img_path, (1280, 1280))
        tiles = tile_image(img_path, os.path.join(tmp, "tiles"), tile_size=640)
        t = tiles[0]
        assert isinstance(t, TileInfo)
        assert t.tile_size == 640
        assert t.row == 0 and t.col == 0
        assert t.x_offset == 0 and t.y_offset == 0


def test_tile_second_column_offset():
    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "scene.jpg")
        make_test_image(img_path, (1280, 640))
        tiles = tile_image(img_path, os.path.join(tmp, "tiles"), tile_size=640)
        assert len(tiles) == 2
        col1 = next(t for t in tiles if t.col == 1)
        assert col1.x_offset == 640


def test_tile_pixel_content():
    with tempfile.TemporaryDirectory() as tmp:
        img = Image.new("RGB", (1280, 1280))
        # Top-left quadrant = red, top-right = green
        for x in range(640):
            for y in range(640):
                img.putpixel((x, y), (255, 0, 0))
                img.putpixel((x + 640, y), (0, 255, 0))
        img_path = os.path.join(tmp, "test.png")
        img.save(img_path)
        tiles = tile_image(img_path, os.path.join(tmp, "tiles"), tile_size=640)
        tl = next(t for t in tiles if t.row == 0 and t.col == 0)
        tr = next(t for t in tiles if t.row == 0 and t.col == 1)
        tl_img = Image.open(tl.tile_path)
        tr_img = Image.open(tr.tile_path)
        assert tl_img.getpixel((0, 0)) == (255, 0, 0)
        assert tr_img.getpixel((0, 0)) == (0, 255, 0)
