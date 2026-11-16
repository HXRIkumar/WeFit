import numpy as np
from PIL import Image
from gpu_service.klein_regions import editing_region, garment_reaches_crop_edge


def test_small_garment_region_keeps_full_photo_coordinates_and_adds_context():
    labels = np.zeros((768, 576), np.uint8); labels[180:380, 200:380] = 3
    region = editing_region(labels, "upper", "auto")
    assert not region.full_frame
    assert region.box[0] < 200 and region.box[1] < 180
    assert region.box[2] > 380 and region.box[3] > 380
    assert region.size[0]*region.size[1] < 576*768
    source = Image.new("RGB", (576, 768), "green")
    result = region.composite(source, Image.new("RGB", region.size, "blue"))
    assert result.getpixel((100, 40)) == source.getpixel((100, 40))
    assert result.getpixel((280, 280)) == (0, 0, 255)


def test_a_large_or_uncertain_region_uses_the_whole_frame():
    assert editing_region(np.full((768, 576), 3, np.uint8), "upper", "auto").full_frame
    assert editing_region(np.zeros((768, 576), np.uint8), "upper", "auto").full_frame


def test_a_new_garment_touching_an_internal_crop_edge_requests_full_render():
    labels = np.zeros((768, 576), np.uint8); labels[180:380, 200:380] = 3
    region = editing_region(labels, "upper", "auto")
    assert not garment_reaches_crop_edge(labels, "upper", region)
    x0, y0, x1, y1 = region.box
    labels[y0:y1, x0:x0+3] = 3
    assert garment_reaches_crop_edge(labels, "upper", region)
