import cv2
import som

import numpy as np
import supervision as sv


class Visualizer:

    def __init__(
        self,
        line_thickness: int = 2,
        mask_opacity: float = 0.1,
        text_scale: float = 0.6
    ) -> None:
        self.box_annotator = sv.BoundingBoxAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            thickness=line_thickness)
        self.mask_annotator = sv.MaskAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            opacity=mask_opacity)
        self.polygon_annotator = sv.PolygonAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            thickness=line_thickness)
        self.label_annotator = sv.LabelAnnotator(
            color=sv.Color.black(),
            text_color=sv.Color.white(),
            color_lookup=sv.ColorLookup.INDEX,
            text_position=sv.Position.CENTER_OF_MASS,
            text_scale=text_scale)

    def visualize(
        self,
        image: np.ndarray,
        detections: sv.Detections,
        with_box: bool,
        with_mask: bool,
        with_polygon: bool,
        with_label: bool
    ) -> np.ndarray:
        annotated_image = image.copy()
        if with_box:
            annotated_image = self.box_annotator.annotate(
                scene=annotated_image, detections=detections)
        if with_mask:
            annotated_image = self.mask_annotator.annotate(
                scene=annotated_image, detections=detections)
        if with_polygon:
            annotated_image = self.polygon_annotator.annotate(
                scene=annotated_image, detections=detections)
        if with_label:
            labels = list(map(str, range(len(detections))))
            annotated_image = self.label_annotator.annotate(
                scene=annotated_image, detections=detections, labels=labels)
        return annotated_image


def refine_mask(
    mask: np.ndarray,
    area_threshold: float,
    mode: str = 'islands'
) -> np.ndarray:
    """
    Refines a mask by removing small islands or filling small holes based on area
    threshold.

    Parameters:
        mask (np.ndarray): Input binary mask.
        area_threshold (float): Threshold for relative area to remove or fill features.
        mode (str): Operation mode ('islands' for removing islands, 'holes' for filling
                    holes).

    Returns:
        np.ndarray: Refined binary mask.
    """
    mask = np.uint8(mask * 255)
    operation = cv2.RETR_EXTERNAL if mode == 'islands' else cv2.RETR_CCOMP
    contours, _ = cv2.findContours(
        mask, operation, cv2.CHAIN_APPROX_SIMPLE
    )
    total_area = cv2.countNonZero(mask) if mode == 'islands' else mask.size

    for contour in contours:
        area = cv2.contourArea(contour)
        relative_area = area / total_area
        if relative_area < area_threshold:
            cv2.drawContours(
                image=mask,
                contours=[contour],
                contourIdx=-1,
                color=(0 if mode == 'islands' else 255),
                thickness=-1
            )

    return np.where(mask > 0, 1, 0).astype(bool)


def filter_masks_by_relative_area(
    masks: np.ndarray,
    min_relative_area: float = 0.02,
    max_relative_area: float = 1.0
) -> np.ndarray:
    """
    Filters out masks based on their relative area.

    Parameters:
        masks (np.ndarray): A 3D numpy array where each slice along the third dimension
            represents a mask.
        min_relative_area (float): Minimum relative area threshold for keeping a mask.
        max_relative_area (float): Maximum relative area threshold for keeping a mask.

    Returns:
        np.ndarray: A 3D numpy array of filtered masks.
    """
    mask_areas = masks.sum(axis=(1, 2))
    total_area = masks.shape[1] * masks.shape[2]
    relative_areas = mask_areas / total_area
    min_area_filter = relative_areas >= min_relative_area
    max_area_filter = relative_areas <= max_relative_area
    return masks[min_area_filter & max_area_filter]


def postprocess_masks(
    detections: sv.Detections,
    area_threshold: float = 0.01,
    min_relative_area: float = 0.01,
    max_relative_area: float = 1.0,
    iou_threshold: float = 0.9
) -> sv.Detections:
    """
    Post-processes the masks of detection objects by removing small islands and filling
    small holes.

    Parameters:
        detections (sv.Detections): Detection objects to be filtered.
        area_threshold (float): Threshold for relative area to remove or fill features.
        min_relative_area (float): Minimum relative area threshold for detections.
        max_relative_area (float): Maximum relative area threshold for detections.
        iou_threshold (float): The IoU threshold above which masks will be considered as
            overlapping.

    Returns:
        np.ndarray: Post-processed masks.
    """
    masks = detections.mask.copy()
    for i in range(len(masks)):
        masks[i] = refine_mask(
            mask=masks[i],
            area_threshold=area_threshold,
            mode='islands'
        )
        masks[i] = refine_mask(
            mask=masks[i],
            area_threshold=area_threshold,
            mode='holes'
        )
    masks = filter_masks_by_relative_area(
        masks=masks,
        min_relative_area=min_relative_area,
        max_relative_area=max_relative_area)
    masks = som.mask_non_max_suppression(
        masks=masks,
        iou_threshold=iou_threshold)

    return sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),
        mask=masks
    )
