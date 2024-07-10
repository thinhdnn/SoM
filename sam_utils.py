import numpy as np
import supervision as sv

from segment_anything.modeling.sam import Sam
from segment_anything import SamPredictor, SamAutomaticMaskGenerator


def sam_inference(
    image: np.ndarray,
    model: Sam
) -> sv.Detections:
    mask_generator = SamAutomaticMaskGenerator(model)
    result = mask_generator.generate(image=image)
    return sv.Detections.from_sam(result)


def sam_interactive_inference(
    image: np.ndarray,
    mask: np.ndarray,
    model: Sam
) -> sv.Detections:
    predictor = SamPredictor(model)
    predictor.set_image(image)
    masks = []
    for polygon in sv.mask_to_polygons(mask.astype(bool)):
        random_point_indexes = np.random.choice(polygon.shape[0], size=5, replace=True)
        input_point = polygon[random_point_indexes]
        input_label = np.ones(5)
        mask = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )[0][0]
        masks.append(mask)
    masks = np.array(masks, dtype=bool)
    return sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),
        mask=masks
    )
