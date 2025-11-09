# Import

import numpy as np
import pandas as pd
from itertools import product
from typing import Tuple, Set, List, Dict, Any, Optional
from tqdm import tqdm
from skimage.measure import regionprops
import skimage.measure
sk_label = skimage.measure.label
from scipy.ndimage import label, generate_binary_structure, binary_erosion, distance_transform_edt, find_objects
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation
import cv2
import csv
import sqlite3
import os
import pickle
import math
import zarr

try:
    import anndata as ad
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    print("Warning: AnnData not available. Install with: pip install anndata")

# Source code functions

## Cell surface precomputation

def get_bounding_boxes_3d(mask_3d: np.ndarray, unique_ids: set) -> dict:
    z, y, x = np.nonzero(mask_3d)
    cell_ids = mask_3d[z, y, x]
    df = pd.DataFrame({'cell_id': cell_ids, 'z': z, 'y': y, 'x': x})
    bbox = {}
    grouped = df.groupby('cell_id')
    for cell_id, group in grouped:
        if cell_id == 0:
            continue
        minz, maxz = group['z'].min(), group['z'].max() + 1
        miny, maxy = group['y'].min(), group['y'].max() + 1
        minx, maxx = group['x'].min(), group['x'].max() + 1
        bbox[cell_id] = (slice(minz, maxz), slice(miny, maxy), slice(minx, maxx))
    return bbox

def compute_bounding_box_with_halo(
    surface_a: np.ndarray,
    max_distance_um: float,
    voxel_size_um: tuple
) -> Tuple[slice, slice, slice]:
    z_coords, y_coords, x_coords = np.where(surface_a)
    
    if len(z_coords) == 0:
        return None
    
    min_z, max_z = z_coords.min(), z_coords.max() + 1
    min_y, max_y = y_coords.min(), y_coords.max() + 1
    min_x, max_x = x_coords.min(), x_coords.max() + 1
    
    pad_z = math.ceil(max_distance_um / voxel_size_um[0])
    pad_y = math.ceil(max_distance_um / voxel_size_um[1])
    pad_x = math.ceil(max_distance_um / voxel_size_um[2])
    
    min_z_pad = max(0, min_z - pad_z)
    max_z_pad = max_z + pad_z + 1
    min_y_pad = max(0, min_y - pad_y)
    max_y_pad = max_y + pad_y + 1
    min_x_pad = max(0, min_x - pad_x)
    max_x_pad = max_x + pad_x + 1
    
    return (slice(min_z_pad, max_z_pad), 
            slice(min_y_pad, max_y_pad), 
            slice(min_x_pad, max_x_pad))

def global_surface_26n(mask_3d: np.ndarray) -> np.ndarray:
    print("Computing global surface mask...")
    
    structure = generate_binary_structure(3, 3)  # 26-connectivity
    binary_mask = (mask_3d > 0).astype(bool)
    eroded = binary_erosion(binary_mask, structure=structure)
    global_surface = binary_mask & ~eroded
    
    print(f"Global surface mask computed: {global_surface.sum()} surface voxels")
    return global_surface

def all_cell_bboxes(mask_3d: np.ndarray) -> Dict[int, Tuple[slice, slice, slice]]:
    print("Computing bounding boxes for all cells...")
    
    unique_ids = set(np.unique(mask_3d))
    unique_ids.discard(0)
    
    bboxes = get_bounding_boxes_3d(mask_3d, unique_ids)
    
    print(f"Bounding boxes computed for {len(bboxes)} cells")
    return bboxes

def precompute_global_surface_and_halo_bboxes(
    mask_3d: np.ndarray, 
    max_distance_um: float,
    voxel_size_um: tuple
) -> Tuple[np.ndarray, Dict[int, Tuple[slice, slice, slice]]]:
    print("Pre-computing global surface and halo-extended bounding boxes...")
    
    # Step 1: Compute global surface mask once
    global_surface = global_surface_26n(mask_3d)
    
    # Step 2: Get all bounding boxes
    all_bboxes = all_cell_bboxes(mask_3d)
    
    # Step 3: Precompute halo-extended bounding boxes
    print("Pre-computing halo-extended bounding boxes...")
    all_bboxes_with_halo = {}
    
    # Calculate halo padding
    pad_z = math.ceil(max_distance_um / voxel_size_um[0])
    pad_y = math.ceil(max_distance_um / voxel_size_um[1])
    pad_x = math.ceil(max_distance_um / voxel_size_um[2])
    
    for cell_id, bbox in all_bboxes.items():
        slice_z, slice_y, slice_x = bbox
        
        # Create extended bounding box with halo
        z_start = max(0, slice_z.start - pad_z)
        z_stop = min(mask_3d.shape[0], slice_z.stop + pad_z)
        y_start = max(0, slice_y.start - pad_y)
        y_stop = min(mask_3d.shape[1], slice_y.stop + pad_y)
        x_start = max(0, slice_x.start - pad_x)
        x_stop = min(mask_3d.shape[2], slice_x.stop + pad_x)
        
        all_bboxes_with_halo[cell_id] = (slice(z_start, z_stop), slice(y_start, y_stop), slice(x_start, x_stop))
    
    print(f"Pre-computed halo-extended bounding boxes for {len(all_bboxes_with_halo)} cells")
    print(f"Using only halo-extended bboxes for all operations")
    
    return global_surface, all_bboxes_with_halo, all_bboxes