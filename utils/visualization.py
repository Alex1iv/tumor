import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# color setup
HU_CLIM = (-1000.0, 300)


def show_candidate(
    dataset,
    ndx=None,
    show_patch=True,
    figsize=(18, 12),
    cmap="gray", # cmap="viridis", #"turbo"
    plot_counter=None, 
    PATH_FIGURES=None
):
    """
    Visualize one candidate from a LunaDataset. 
    
    The full CT views are displayed in the native NumPy coordinate system (I, R, C), 
    with the candidate center    marked correctly.

    The candidate patch is displayed using raw HU values,
    not the normalized tensor returned by LunaDataset.

    Parameters
    ----------
    dataset : LunaDataset
        Dataset containing the candidates to visualize.

    ndx : int or None
        Index of the candidate in dataset.candidates_df.
        If None, the first positive candidate is selected.

    show_patch : bool. Defaults to True.
        If True, also display the extracted 3-D candidate patch.

    figsize : tuple. Defaults to (18, 12)
        Matplotlib figure size.
    
    cmap (str, optional): 
        Color palette. Defaults to "gray".
    
    plot_counter (int, optional)
        Figure number. Defaults to None.
        
    PATH_FIGURES (str, optional)
        path to the images directory
    """

    
    # Select candidate
    if len(dataset) == 0:
        raise ValueError("Dataset contains no candidates.")

    if ndx is None:
        positive_indices = np.flatnonzero(
            dataset.candidates_df["class"].to_numpy() == 1
        )

        if len(positive_indices) > 0:
            ndx = int(positive_indices[0])
        else:
            print("Warning: no positive samples found; using first candidate.")
            ndx = 0

    if ndx < 0 or ndx >= len(dataset):
        raise IndexError(
            f"Candidate index {ndx} is outside dataset range [0, {len(dataset) - 1}]."
        )

    # Get candidate metadata
    row = dataset.candidates_df.iloc[ndx]

    series_uid = row["series_uid"]
    label = int(row["class"])

    center_xyz = np.array([row["coordX"], row["coordY"], row["coordZ"]], dtype=np.float64)

    
    # Load CT 
    ct = dataset._get_ct(series_uid)
    
    # Convert XYZ -> IRC
    center_irc = ct.xyz_to_irc(center_xyz)

    center_i = int(round(center_irc[0]))
    center_r = int(round(center_irc[1]))
    center_c = int(round(center_irc[2]))
    
    # Extract RAW HU patch through Dataset
    raw_patch, patch_center_irc = ct.get_raw_candidate(center_xyz, dataset.width_irc)

    assert np.allclose(center_irc, patch_center_irc)
    
    # Print information
    print(f"Series UID : {series_uid}")
    print(f"Candidate  : {ndx}")
    print(f"Class      : {label}")
    print(f"Diameter   : {row['diameter_mm']:.3f} mm")
    print(f"XYZ        : {center_xyz}")
    print(f"IRC        : {center_irc}")
    print(f"CT shape   : {ct.hu_array.shape}")
    print(f"Patch shape: {raw_patch.shape}")

    # Check that center is inside CT
    I, R, C = ct.hu_array.shape

    if not (
        0 <= center_i < I
        and 0 <= center_r < R
        and 0 <= center_c < C
    ):
        raise ValueError(
            f"Candidate center {center_irc} is outside "
            f"CT dimensions {ct.hu_array.shape}"
        )
        
    # Plot
    fig, ax = plt.subplots(2,3, figsize=figsize)

    # Three full CT views around the candidate 
    # axial. coords (x, y): (C, R)
    ax[0, 0].imshow(
        ct.hu_array[center_i], clim=HU_CLIM, cmap=cmap #, origin="upper"
        
    )

    ax[0, 0].axvline(center_c, linewidth=1)
    ax[0, 0].axhline(center_r,  linewidth=1)
    
    ax[0, 0].set_xlabel("C")
    ax[0, 0].set_ylabel("R")
    ax[0, 0].set_title(f"Axial: I={center_i}", fontsize=14)
    #ax[0, 0].invert_yaxis()
    
    # CORONAL / ROW. Image shape = (I, C); coords (x, y): (C, I)
    ax[0, 1].imshow(
        ct.hu_array[:, center_r, :], clim=HU_CLIM, cmap=cmap, aspect="auto"
    )
    
    ax[0, 1].axvline(center_c, linewidth=1) # C - column
    ax[0, 1].axhline(center_i, linewidth=1) # I - index
    
    ax[0, 1].set_xlabel("C")
    ax[0, 1].set_ylabel("I")
    ax[0, 1].set_title(f"Coronal: R={center_r}", fontsize=14)
    ax[0, 1].invert_yaxis()
    
    # SAGITTAL / COLUMN. Image shape = (I, R); coords (x, y): (R, I)  
    ax[0, 2].imshow(
        ct.hu_array[:, :, center_c], 
        clim=HU_CLIM, cmap=cmap, aspect="auto"
    )
    ax[0, 2].axvline(center_r, linewidth=1) # R
    ax[0, 2].axhline(center_i,  linewidth=1) # I
    
    ax[0, 2].set_xlabel("R")
    ax[0, 2].set_ylabel("I")
    ax[0, 2].set_title(f"Sagittal: C={center_c}", fontsize=14)
    ax[0, 2].invert_yaxis()
    
    # Candidate patch
    if show_patch:

        patch_i = raw_patch.shape[0] // 2
        patch_r = raw_patch.shape[1] // 2
        patch_c = raw_patch.shape[2] // 2

        # Patch axial
        ax[1, 0].imshow(
            raw_patch[patch_i],
            clim=HU_CLIM,
            cmap=cmap
        )
        ax[1, 0].axvline(patch_c, linewidth=1)
        ax[1, 0].axhline(patch_r, linewidth=1)

        ax[1, 0].set_xlabel("C")
        ax[1, 0].set_ylabel("R")
        ax[1, 0].set_title(f"Patch axial: I={patch_i}", fontsize=14)
        ax[1, 0].invert_yaxis()
        
        # Patch coronal
        ax[1, 1].imshow(
            raw_patch[:, patch_r, :],
            clim=HU_CLIM,
            cmap=cmap,
            aspect="auto"
        )
        ax[1, 1].axvline(patch_c, linewidth=1)
        ax[1, 1].axhline(patch_i, linewidth=1)

        ax[1, 1].set_xlabel("C")
        ax[1, 1].set_ylabel("I")
        ax[1, 1].set_title(f"Patch coronal: R={patch_r}", fontsize=14)
        ax[1, 1].invert_yaxis()
        
        # Patch sagittal
        ax[1, 2].imshow(
            raw_patch[:, :, patch_c],
            clim=HU_CLIM,
            cmap=cmap,
            aspect="auto"
        )
        
        ax[1, 2].axvline(patch_r, linewidth=1)
        ax[1, 2].axhline(patch_i, linewidth=1)

        ax[1, 2].set_xlabel("R")
        ax[1, 2].set_ylabel("I")
        ax[1, 2].set_title(f"Patch sagittal: C={patch_c}", fontsize=14)
        ax[1, 2].invert_yaxis()

    plt.tight_layout()
    
    if plot_counter:

        # Create the directory if it does not exist
        Path(PATH_FIGURES).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(PATH_FIGURES, f'fig_{plot_counter}.svg'),format="svg", bbox_inches="tight", transparent=True);