# import numpy as np
# from collections import namedtuple
# import gzip
# import SimpleITK as sitk
# import numpy as np
# from diskcache import FanoutCache, Disk
# from diskcache.core import BytesType, MODE_BINARY, BytesIO

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import SimpleITK as sitk
from joblib import Memory

# Persistent cache
memory = Memory(
    location="../data/cache/candidate_info",
    verbose=0
)


@memory.cache
def augment_candidates(
    annotations: pd.DataFrame,
    candidates: pd.DataFrame,
    match_fraction: float = 0.25 
):
    """
    Augment LUNA16 candidates with nodule diameter information. 
    
    For every candidate, annotations with the same seriesuid are searched.
    If a candidate is sufficiently close to an annotation, the annotation's
    diameter_mm is assigned to that candidate.

    Otherwise, diameter_mm is set to 0.0.

    Matching criterion for all three spatial dimensions:
        abs(candidate_coord[i] - annotation_coord[i]) <= annotation_diameter_mm / 4

    
    Parameters
    ----------
    annotations: (pd.DataFrame) Expected columns:
        [series_uid, coordX, coordY, coordZ, diameter_mm]

    candidates: (pd.DataFrame) Expected columns:
        [series_uid, coordX, coordY, coordZ, class]
    
    match_fraction:(float) Default=1/4 
        Fraction of the annotation diameter used as the coordinate-wise matching threshold.
       
        
    Returns
    -------
    pd.DataFrame
   
       diameter_mm is the matched annotation diameter or 0.0 if
        no matching annotation is found.
    """

    annotation_columns = {"series_uid", "coordX", "coordY", "coordZ", "diameter_mm"}
    candidate_columns = { "series_uid", "coordX", "coordY", "coordZ", "class"}

    missing_annotations = annotation_columns - set(annotations.columns)
    missing_candidates = candidate_columns - set(candidates.columns)

    if missing_annotations:
        raise ValueError(
            f"annotations is missing columns: {missing_annotations}"
        )

    if missing_candidates: 
        raise ValueError(
            f"candidates is missing columns: {missing_candidates}"
        )
        
    result = candidates.copy()
    annotations = annotations.copy()

    coord_columns  = ["coordX", "coordY", "coordZ"]

    # 3. Prepare output feature. By default, every candidate is considered unmatched.
    result["diameter_mm"] = np.float32(0.0)  
    
    # Process one CT series at a time
    annotation_groups = annotations.groupby("series_uid", sort=False)

    for series_uid, candidate_group  in result.groupby("series_uid", sort=False):

        # Skip this series if it has no reference annotations.
        if series_uid not in annotation_groups.groups:
            continue

        ann_group = annotation_groups.get_group(series_uid)

        candidate_indices = candidate_group.index
        candidate_xyz = candidate_group[coord_columns].to_numpy(dtype=np.float64)

        annotation_xyz = ann_group[coord_columns].to_numpy(dtype=np.float64)
        annotation_diameters = ann_group["diameter_mm"].to_numpy(dtype=np.float64)
        
        # KD-tree
        tree = cKDTree(annotation_xyz)

        # The original matching rule defines a cube:
        #
        # |dx| <= diameter / 4
        # |dy| <= diameter / 4
        # |dz| <= diameter / 4
        #
        # A KD-tree uses Euclidean distance, so we first retrieve all potentially matching annotations 
        # using the largest possible distance from the center to a corner of such a cube.

        max_search_radius = (
            np.sqrt(3.0) * annotation_diameters.max() * match_fraction
        )

        possible_matches = tree.query_ball_point(
            candidate_xyz,  r=max_search_radius
        )

        # Check only annotations returned by the KD-tree.
        for candidate_pos, ann_positions in enumerate(possible_matches):
            if not ann_positions:
                continue

            candidate_point = candidate_xyz[candidate_pos]

            ann_positions = np.asarray(ann_positions, dtype=np.int64)
            
            nearby_xyz = annotation_xyz[ann_positions]
            nearby_diameters = annotation_diameters[ann_positions]

            coordinate_differences = np.abs(nearby_xyz - candidate_point)
            threshold = nearby_diameters * match_fraction
            
            matches = np.all(coordinate_differences <= threshold[:, None], axis=1)

            if np.any(matches):

                # The original code stops at the first matching annotation.
                # If several annotations match, we assign the first one.
                first_match_position = np.flatnonzero(matches)[0]

                result.loc[candidate_indices[candidate_pos], "diameter_mm"
                           ] = nearby_diameters[first_match_position]

    return result



def find_ct_files(data_dir):
    """
    Find all LUNA16 .mhd/.raw pairs.

    Returns
    -------
    list[dict]
        Information about available CT volumes.
    """

    data_dir = Path(data_dir)

    ct_files = []

    for mhd_path in sorted(data_dir.glob("subset*/*.mhd")):
        raw_path = mhd_path.with_suffix(".raw")

        if not raw_path.exists():
            print(f"Warning: missing RAW file for {mhd_path}")
            continue

        series_uid = mhd_path.stem

        ct_files.append({
            "series_uid": series_uid,
            "mhd_path": mhd_path,
            "raw_path": raw_path
        })

    return ct_files



class CTVolume:
    """Represents one LUNA16 CT volume.
    The CT image is stored as a NumPy array with dimensions:
        (I, R, C) = (slice, row, column)

    Physical coordinates are expressed in:
        (X, Y, Z) millimeters

    The conversion between the two coordinate systems is performed
    using the spatial metadata stored in the .mhd header.
    """

    def __init__(self, series_uid: str, data_dir: str):

        self.series_uid = series_uid
        self.data_dir = Path(data_dir)
        
        # Find the corresponding .mhd file
        mhd_files = list(self.data_dir.glob(f"subset*/{series_uid}.mhd"))

        if not mhd_files:
            raise FileNotFoundError(
                f"No .mhd file found for series_uid={series_uid}"
            )

        if len(mhd_files) > 1:
            raise RuntimeError(
                f"Multiple .mhd files found for {series_uid}: {mhd_files}"
            )

        self.mhd_path = mhd_files[0]

        # Corresponding .raw file
        self.raw_path = self.mhd_path.with_suffix(".raw")

        if not self.raw_path.exists():
            raise FileNotFoundError(f"Expected .raw file not found: {self.raw_path}")

        # Read CT using SimpleITK package
        self.image = sitk.ReadImage(str(self.mhd_path)) 

        # SimpleITK returns a NumPy array as: (Z, Y, X) which we call: (I, R, C)
        #.astype(np.float32, copy=False) overloads cache
        # We deliberately do NOT convert the entire CT volume to a NumPy array
        # here. Only the small candidate patch is converted to NumPy in
        # get_raw_candidate(). This substantially reduces RAM usage.
        
        #self.hu_array = sitk.GetArrayFromImage(self.image).astype(np.float32, copy=False)

        # Clip HU values
        #np.clip(self.hu_array, -1000.0, 1000.0, out=self.hu_array)

        
        # Spatial metadata
        self.origin_xyz = np.asarray(
            self.image.GetOrigin(), dtype=np.float64)

        self.spacing_xyz = np.asarray(
            self.image.GetSpacing(), dtype=np.float64)

        self.direction = np.asarray(
            self.image.GetDirection(), dtype=np.float64).reshape(3, 3)

    # Coordinate conversion
    def xyz_to_irc(self, xyz):
        """
        Convert physical LPS coordinates (X,Y,Z) in mm 
        to NumPy array coordinates (I,R,C).

        Parameters
        ----------
        xyz : array-like, shape (3,)
            Physical coordinate in millimeters.

        Returns
        -------
        np.ndarray, shape (3,)
            Continuous array coordinate (I,R,C).
        """

        xyz = np.asarray(xyz, dtype=np.float64)

        if xyz.shape != (3,):
            raise ValueError(f"Expected xyz shape (3,), got {xyz.shape}")

        # SimpleITK's TransformPhysicalPointToContinuousIndex
        # returns index coordinates in (X,Y,Z) / (C,R,I) order.
        continuous_index_xyz = np.asarray(
            self.image.TransformPhysicalPointToContinuousIndex(
                tuple(xyz)), dtype=np.float64)

        # NumPy array order is (I,R,C), therefore reverse.
        # .copy() is important because [::-1] otherwise creates 
        # an array with negative strides.
        irc = continuous_index_xyz[::-1].copy()

        return irc

    
    
    def get_raw_candidate(self, center_xyz:np.ndarray, width_irc:tuple):
        """Gets physical candidate center, calculate nodule boundaries 
        and extract a 3D CT patch centered at a physical XYZ coordinate.

        Args:
            center_xyz (_ndarray_): Candidate center in physical LPS coordinates (mm)
            width_irc (tuple): Patch size in NumPy array coordinates: (I, R, C).

        Returns:
            ct_chunk (np.ndarray):  CT patch with shape approximately width_irc.

            center_irc (np.ndarray)
                Continuous candidate center in (I, R, C).
        """  
        width_irc = tuple(int(x) for x in width_irc)

        if len(width_irc) != 3:
            raise ValueError(f"width_irc must contain 3 values, got {width_irc}")

        if any(x <= 0 for x in width_irc):
            raise ValueError(f"width_irc must contain positive values, got {width_irc}")

      
        center_irc = self.xyz_to_irc(center_xyz)

        slice_list = []
        
        # SimpleITK image size is expressed in (X, Y, Z) order,
        # while our NumPy/IRC convention is (I, R, C).
        image_size_irc = tuple(self.image.GetSize())[::-1]
        

        for axis, center_val in enumerate(center_irc):

            start_ndx = int(round(center_val - width_irc[axis] / 2))

            end_ndx = start_ndx + width_irc[axis]

            # Boundary handling
            if start_ndx < 0:
                start_ndx = 0
                end_ndx = width_irc[axis]

            if end_ndx > image_size_irc[axis]:
                end_ndx = image_size_irc[axis]
                start_ndx = image_size_irc[axis] - width_irc[axis]
                
            slice_list.append( slice(start_ndx, end_ndx ) )

        #ct_chunk = self.hu_array[tuple(slice_list)]
         # Convert NumPy/IRC coordinates to SimpleITK's XYZ order.
        start_irc = np.array( [s.start for s in slice_list], dtype=np.int64)

        size_irc = np.array([s.stop - s.start for s in slice_list], dtype=np.int64)

        start_xyz = tuple(start_irc[::-1].tolist())
        size_xyz = tuple(size_irc[::-1].tolist())

        # Extract only the required 3D region from the CT.
        ct_image = sitk.RegionOfInterest(self.image, size=size_xyz, index=start_xyz)

        # SimpleITK returns a NumPy array as: (Z, Y, X) which we call: (I, R, C)
        # Convert only the small candidate patch to float32.
        ct_chunk = sitk.GetArrayFromImage(ct_image).astype(np.float32, copy=False)

        # Clip HU values
        np.clip(ct_chunk, -1000.0, 1000.0, out=ct_chunk)
        
        return ct_chunk, center_irc
    
    def get_raw_slice(self, axis: int, index: int):
        """Gets one raw HU slice from the CT volume.

        Args:
            axis (int): NumPy/IRC axis:
                0 = I (axial)
                1 = R (coronal)
                2 = C (sagittal)

            index (int): Slice index along the selected axis.

        Returns:
            slice_array (np.ndarray):
                2D CT slice in the native NumPy coordinate system.
        """

        # SimpleITK image size is expressed in (X, Y, Z) order,
        # while our NumPy/IRC convention is (I, R, C).
        image_size_irc = tuple(self.image.GetSize())[::-1]

        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")

        if index < 0 or index >= image_size_irc[axis]:
            raise IndexError(
                f"Slice index {index} is outside axis {axis} "
                f"range [0, {image_size_irc[axis] - 1}]"
            )

        # Create a region containing one slice along the requested axis.
        start_irc = [0, 0, 0]
        size_irc = list(image_size_irc)

        start_irc[axis] = index
        size_irc[axis] = 1

        # Convert NumPy/IRC coordinates to SimpleITK's XYZ order.
        start_xyz = tuple(start_irc[::-1])
        size_xyz = tuple(size_irc[::-1])

        slice_image = sitk.RegionOfInterest(
            self.image,
            size=size_xyz,
            index=start_xyz
        )

        # SimpleITK returns a NumPy array as: (Z, Y, X) which we call: (I, R, C)
        slice_array = sitk.GetArrayFromImage(
            slice_image
        ).astype(np.float32, copy=False)

        # Clip HU values
        np.clip(
            slice_array,
            -1000.0,
            1000.0,
            out=slice_array
        )

        # Remove the dimension of size 1.
        slice_array = np.squeeze(slice_array, axis=axis)

        return slice_array


import logging
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class LunaDataset(Dataset):

    def __init__(
        self,
        candidates_df: pd.DataFrame,
        data_dir: str = "../data/raw",
        width_irc=(32, 48, 48),
        series_uid=None,
        normalize=True
    ):
        """
        Parameters
        ----------
        candidates_df : pd.DataFrame
            Augmented LUNA16 candidate dataframe.

        data_dir : str
            Directory containing subset*/<series_uid>.mhd/.raw.

        width_irc : tuple
            Patch dimensions in (I, R, C).

        series_uid : str or None
            If specified, use only candidates from this CT series.

        normalize : bool
            If True, map HU values from [-1000, 1000]
            to approximately [-1, 1].
        """

        super().__init__()

        self.data_dir = Path(data_dir)
        self.width_irc = tuple(width_irc)
        self.normalize = normalize

        # Validate DataFrame
        required_columns = {"series_uid", "coordX", "coordY", "coordZ", "class", "diameter_mm" }

        missing = required_columns - set(candidates_df.columns)

        if missing:
            raise ValueError(f"candidates_df is missing columns: {missing}")


        # Keep only required information
        self.candidates_df = candidates_df[["series_uid", "coordX", "coordY", "coordZ", "class", "diameter_mm"]].copy()


        # Optional series filtering
        if series_uid is not None:
            self.candidates_df = self.candidates_df[
                self.candidates_df["series_uid"] == series_uid
            ].copy()


        if len(self.candidates_df) == 0:
            raise ValueError("LunaDataset contains no samples.")

        # Reset index
        self.candidates_df = self.candidates_df.reset_index(drop=True)

        # One-CT RAM cache
        self._cached_series_uid = None
        self._cached_ct = None
    
    # Dataset length
    def __len__(self):
        return len(self.candidates_df)

    
    # CT loading
    def _get_ct(self, series_uid):
        
        if self._cached_series_uid != series_uid:
            # Release the previous CT before loading a new one.
            self._cached_ct = None
            self._cached_series_uid = None
            
            import gc
            gc.collect()
        
            self._cached_ct = CTVolume(
                series_uid=series_uid,
                data_dir=self.data_dir
            )

            self._cached_series_uid = series_uid

        return self._cached_ct

    
    # HU normalization
    def _normalize_hu(self, ct_chunk):

        if not self.normalize:
            return ct_chunk

        # HU was clipped to [-1000, 1000].
        #
        # Map:
        #
        # -1000 -> -1
        #     0 ->  0
        #  1000 -> +1
        #
        return ct_chunk / 1000.0

    
    # Get one sample
    def __getitem__(self, ndx):

        row = self.candidates_df.iloc[ndx]

        series_uid = row["series_uid"]
        center_xyz = np.array(
            [row["coordX"], row["coordY"], row["coordZ"]], dtype=np.float64
        )

        # Load CT volume
        ct = self._get_ct(series_uid)

        # Extract 3-D CT patch
        ct_patch, center_irc = ct.get_raw_candidate(center_xyz, self.width_irc)
        
        # Normalize HU
        ct_patch = self._normalize_hu(ct_patch.astype(np.float32, copy=False))

        # Cast NumPy array to Torch and Add channel dimension: (I, R, C) → (1, I, R, C)
        candidate_t = torch.from_numpy(ct_patch).unsqueeze(0)
        #.to(torch.float32, copy=False).unsqueeze(0)


        # Classification target
        label_t = torch.tensor(row["class"], dtype=torch.long)

        # Compute candidate center in voxel coordinates
        center_irc_t = torch.tensor(center_irc, dtype=torch.float32)

        return (
            candidate_t,
            label_t,
            series_uid,
            center_irc_t
        )