import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)


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
    raw_patch, patch_center_irc = ct.get_raw_candidate(
        center_xyz, 
        dataset.width_irc
    )

    assert np.allclose(center_irc, patch_center_irc)
    
    # Print information
    print(f"Series UID : {series_uid}")
    print(f"Candidate  : {ndx}")
    print(f"Class      : {label}")
    print(f"Diameter   : {row['diameter_mm']:.3f} mm")
    print(f"XYZ        : {center_xyz}")
    print(f"IRC        : {center_irc}")
    print(f"Patch shape: {raw_patch.shape}")

    # Check that center is inside CT
    I, R, C = tuple(ct.image.GetSize())[::-1]

    if not (
        0 <= center_i < I
        and 0 <= center_r < R
        and 0 <= center_c < C
    ):
        raise ValueError(
            f"Candidate center {center_irc} is outside CT dimensions {(I, R, C)}")
    
    # Get the three full CT views around the candidate.
    # Only the required 2-D slices are loaded into NumPy.
    axial_slice = ct.get_raw_slice(axis=0, index=center_i)
    coronal_slice = ct.get_raw_slice(axis=1, index=center_r)
    sagittal_slice = ct.get_raw_slice(axis=2, index=center_c)
    
    # Plot
    fig, ax = plt.subplots(2,3, figsize=figsize)

    # Three full CT views around the candidate 
    # axial. coords (x, y): (C, R)
    ax[0, 0].imshow(axial_slice, clim=HU_CLIM, cmap=cmap) #, origin="upper"   

    ax[0, 0].axvline(center_c, linewidth=1, color='r')
    ax[0, 0].axhline(center_r,  linewidth=1, color='r')
    
    ax[0, 0].set_xlabel("C")
    ax[0, 0].set_ylabel("R")
    ax[0, 0].set_title(f"Axial: I={center_i}", fontsize=14)
    #ax[0, 0].invert_yaxis()
    
    # CORONAL / ROW. Image shape = (I, C); coords (x, y): (C, I)
    ax[0, 1].imshow(coronal_slice, clim=HU_CLIM, cmap=cmap, aspect="auto")
    
    ax[0, 1].axvline(center_c, linewidth=1, color='r') # C - column
    ax[0, 1].axhline(center_i, linewidth=1, color='r') # I - index
    
    ax[0, 1].set_xlabel("C")
    ax[0, 1].set_ylabel("I")
    ax[0, 1].set_title(f"Coronal: R={center_r}", fontsize=14)
    ax[0, 1].invert_yaxis()
    
    # SAGITTAL / COLUMN. Image shape = (I, R); coords (x, y): (R, I)  
    ax[0, 2].imshow(sagittal_slice, clim=HU_CLIM, cmap=cmap, aspect="auto")
    
    ax[0, 2].axvline(center_r, linewidth=1, color='r') # R
    ax[0, 2].axhline(center_i,  linewidth=1, color='r') # I
    
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
        ax[1, 0].axvline(patch_c, linewidth=1, color='r')
        ax[1, 0].axhline(patch_r, linewidth=1, color='r')

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
        ax[1, 1].axvline(patch_c, linewidth=1, color='r')
        ax[1, 1].axhline(patch_i, linewidth=1, color='r')

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
        
        ax[1, 2].axvline(patch_r, linewidth=1, color='r')
        ax[1, 2].axhline(patch_i, linewidth=1, color='r')

        ax[1, 2].set_xlabel("R")
        ax[1, 2].set_ylabel("I")
        ax[1, 2].set_title(f"Patch sagittal: C={patch_c}", fontsize=14)
        ax[1, 2].invert_yaxis()

    plt.tight_layout()
    
    if plot_counter:

        # Create the directory if it does not exist
        Path(PATH_FIGURES).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(PATH_FIGURES, f'fig_{plot_counter}.svg'),format="svg", bbox_inches="tight", transparent=True);
        

def plot_classification_curves(model, val_loader, device, title="Lungs tumor clf"):
    """
    Plot ROC-AUC and Precision-Recall curves for a trained
    binary PyTorch image classifier.

    Args:
        model: trained PyTorch model
        val_loader: DataLoader containing validation images and labels
        device: torch.device("cuda") or torch.device("cpu")
        title: title of the figure

    Returns:
        roc_auc: ROC-AUC score
        average_precision: Average Precision score
    """
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for imgs, labels , _, _ in val_loader:

            imgs = imgs.to(device)

            # Model returns logits: [batch_size, 2]
            outputs = model(imgs)

            # Convert logits to probabilities
            probs = torch.softmax(outputs, dim=1)

            # Probability of positive class (class 1)
            positive_probs = probs[:, 1]

            all_labels.append(labels.cpu())
            all_probs.append(positive_probs.cpu())

    # Combine all batches
    y_true = torch.cat(all_labels).numpy()
    y_score = torch.cat(all_probs).numpy()

    
    # ROC curve
    false_positive_rates, true_positive_rates, roc_thresholds = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    
    # Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)

    average_precision = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))

    # ROC-AUC
    ax[0].plot(
        false_positive_rates,
        true_positive_rates,
        label=f"ROC-AUC = {roc_auc:.3f}"
    )

    ax[0].plot([0, 1], [0, 1], linestyle=":", label="Random classifier")

    ax[0].set_title(f"ROC Curve\n{title}")
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_xlim([0, 1])
    ax[0].set_ylim([0, 1.05])
    ax[0].legend(loc="lower right")
    ax[0].grid(alpha=0.3)

    # Precision-Recall
    ax[1].plot(recall, precision, label=f"AP = {average_precision:.3f}")

    # Baseline = proportion of positive samples
    positive_rate = y_true.mean()

    ax[1].axhline(
        positive_rate, linestyle=":", label=f"Random classifier = {positive_rate:.3f}"
    )

    ax[1].set_title(f"Precision-Recall Curve\n{title}")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_xlim([0, 1])
    ax[1].set_ylim([0, 1.05])
    ax[1].legend(loc="lower left")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision: {average_precision:.4f}")