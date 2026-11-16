"""Fixed, garment-independent instructions for reference-conditioned editing."""
import hashlib

MODEL_REPOSITORY = "mflux-community/flux2-klein-9b-kv-mflux-q4"
MODEL_REVISION = "d144dcb1f70540f581ed99930ee61aac21b5c6d6"
INSTRUCTIONS = {
    "upper": (
        "Replace only the person's upper-body clothing in image 1 with the exact garment shown in image 2. "
        "Reproduce the reference garment's actual cut, color, pattern, material, collar, sleeves, seams and fastenings. "
        "Remove the previous top completely instead of layering the new garment over it. "
        "Keep the same person, face, hair, skin, body proportions, pose, hands, lower-body clothing, background, "
        "camera angle, framing, lighting and photographic grain from image 1. "
        "The result is the original natural photograph with just the reference garment fitted to this person's body."
    ),
    "lower": (
        "Replace only the person's trousers or shorts in image 1 with the exact trousers shown in image 2. "
        "Reproduce the reference garment's actual cut, color, pattern, fabric, waistband, seams, fastenings and leg length. "
        "Fit the trousers to this person's actual waist, hips, legs and standing pose. "
        "Keep the same person, face, hair, skin, body proportions, pose, hands, upper-body clothing, feet, background, "
        "camera angle, framing, lighting and photographic grain from image 1. "
        "The result is the original natural photograph with just the reference trousers fitted on this person."
    ),
}


def instruction_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
