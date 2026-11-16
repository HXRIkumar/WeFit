import numpy as np
from PIL import Image
from gpu_service.photo_preserve import preserve_reference_edit


def test_newly_exposed_room_is_not_rebuilt_from_old_dark_clothing():
    pixels = np.full((120, 80, 3), (110, 180, 150), np.uint8)
    pixels[20:110, 10:70] = (2, 5, 10)
    old = np.zeros((120, 80), np.uint8); old[20:110, 10:70] = 3
    new = np.zeros_like(old); new[20:110, 30:50] = 3
    generated = np.full_like(pixels, (110, 180, 150)); generated[20:110, 30:50] = (140, 170, 210)
    result = preserve_reference_edit(Image.fromarray(pixels), Image.fromarray(generated), old, "upper", new)
    assert result.getpixel((20, 70)) == (110, 180, 150)
    assert result.getpixel((40, 70)) == (140, 170, 210)


def test_uncertain_old_face_label_does_not_punch_a_hole_in_the_new_collar():
    pixels = np.full((120, 100, 3), (110, 180, 150), np.uint8)
    pixels[10:65, 30:70] = (100, 70, 60)
    old = np.zeros((120, 100), np.uint8); old[10:75, 25:75] = 1
    new = np.zeros_like(old); new[10:60, 30:70] = 1; new[65:110, 15:85] = 3
    generated = np.full_like(pixels, (140, 170, 210))
    result = preserve_reference_edit(Image.fromarray(pixels), Image.fromarray(generated), old, "upper", new)
    assert result.getpixel((50, 35)) == (100, 70, 60)
    assert result.getpixel((28, 71)) == (140, 170, 210)


def test_original_pants_are_preserved_except_where_the_new_shirt_extends():
    old = np.full((120, 80), 3, np.uint8); old[50:] = 6
    new = old.copy(); new[50:80] = 3
    result = preserve_reference_edit(Image.new("RGB", (80, 120), "black"),
        Image.new("RGB", (80, 120), "blue"), old, "upper", new)
    assert result.getpixel((40, 65)) == (0, 0, 255)
    assert result.getpixel((40, 100)) == (0, 0, 0)


def test_missed_dark_source_outline_does_not_become_a_background_sample():
    source = np.full((120, 100, 3), (110, 180, 150), np.uint8)
    source[20:110, 23:77] = [2, 5, 10]
    old = np.zeros((120, 100), np.uint8); old[20:110, 30:70] = 3
    new = np.zeros_like(old); new[20:110, 40:60] = 3
    generated = np.full_like(source, (130, 195, 165)); generated[20:110, 40:60] = [140, 170, 210]
    result = preserve_reference_edit(Image.fromarray(source), Image.fromarray(generated), old, "upper", new)
    pixel = np.asarray(result)[70, 25]
    assert np.max(np.abs(pixel.astype(float)-[110, 180, 150])) < 10
    assert result.getpixel((50, 70)) == (140, 170, 210)
