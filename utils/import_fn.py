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
            "raw_path": raw_path}
        )

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
                f"Multiple .mhd files found for {series_uid}: "
                f"{mhd_files}"
            )

        self.mhd_path = mhd_files[0]

        # Corresponding .raw file
        self.raw_path = self.mhd_path.with_suffix(".raw")

        if not self.raw_path.exists():
            raise FileNotFoundError(
                f"Expected .raw file not found: {self.raw_path}"
            )

        # Read CT
        self.image = sitk.ReadImage(str(self.mhd_path))

        # SimpleITK returns a NumPy array as: (Z, Y, X) which we call: (I, R, C)
        self.hu_array = sitk.GetArrayFromImage(
            self.image).astype(np.float32, copy=False)

        
        # Clip HU values
        np.clip(self.hu_array, -1000.0, 1000.0, out=self.hu_array)

        
        # Spatial metadata
        self.origin_xyz = np.asarray(
            self.image.GetOrigin(),   dtype=np.float64)

        self.spacing_xyz = np.asarray(
            self.image.GetSpacing(),  dtype=np.float64)

        self.direction = np.asarray(
            self.image.GetDirection(), dtype=np.float64).reshape(3, 3)

    # Coordinate conversion
    def xyz_to_irc(self, xyz):
        """
        Convert physical LPS coordinates (X,Y,Z) in mm to NumPy array coordinates (I,R,C).

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
            raise ValueError(
                f"Expected xyz shape (3,), got {xyz.shape}"
            )

        # SimpleITK's TransformPhysicalPointToContinuousIndex
        # returns index coordinates in (X,Y,Z) / (C,R,I) order.
        continuous_index_xyz = np.asarray(
            self.image.TransformPhysicalPointToContinuousIndex(
                tuple(xyz)), dtype=np.float64)

        # NumPy array order is (I,R,C), therefore reverse.
        irc = continuous_index_xyz[::-1]

        return irc

    
    # Convert all candidates belonging to this CT
    def add_candidate_coordinates(
        self,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:

        result = candidates.copy()
        xyz = result[["coordX", "coordY", "coordZ"]].to_numpy(dtype=np.float64)

        irc = np.array(
            [self.xyz_to_irc(point) for point in xyz],
            dtype=np.float64)

        result[["centerI", "centerR", "centerC"]] = irc

        return result

# class Ct:
#     def __init__(self, series_uid):
#         mhd_path = glob.glob(
#             'data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid)
#         )[0]

#         ct_mhd = sitk.ReadImage(mhd_path)
#         ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

#         # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
#         # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
#         # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
#         # The upper bound nukes any weird hotspots and clamps bone down
#         ct_a.clip(-1000, 1000, ct_a)

#         self.series_uid = series_uid
#         self.hu_a = ct_a

#         self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
#         self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
#         self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

#     def getRawCandidate(self, center_xyz, width_irc):
#         center_irc = xyz2irc(
#             center_xyz,
#             self.origin_xyz,
#             self.vxSize_xyz,
#             self.direction_a,
#         )

#         slice_list = []
#         for axis, center_val in enumerate(center_irc):
#             start_ndx = int(round(center_val - width_irc[axis]/2))
#             end_ndx = int(start_ndx + width_irc[axis])

#             assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

#             if start_ndx < 0:
#                 # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
#                 #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
#                 start_ndx = 0
#                 end_ndx = int(width_irc[axis])

#             if end_ndx > self.hu_a.shape[axis]:
#                 # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
#                 #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
#                 end_ndx = self.hu_a.shape[axis]
#                 start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

#             slice_list.append(slice(start_ndx, end_ndx))

#         ct_chunk = self.hu_a[tuple(slice_list)]

#         return ct_chunk, center_irc