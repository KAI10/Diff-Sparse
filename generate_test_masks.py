import torch
import pickle

def generate_mask(p, mask_size):
    """
    Generates a missing data mask M using a probabilistic approach.
    For each cell, it is marked as missing with the given probability.

    M[i, j] = 1 if the cell is 'observed'.
    M[i, j] = 0 if the cell is stochastically 'missing'.

    Args:
        p: The probability (0.0 to 1.0) that any given cell will be marked as missing.
        mask_size: Shape of mask will be mask_size x mask_size

    Returns:
        M (mask_size, mask_size) array representing the missing data mask M.
    """
    assert 0 <= p <= 1
    mask = torch.ones(mask_size, mask_size, dtype=torch.int8)
    condition_to_mask = torch.rand(mask_size, mask_size) < p
    mask[condition_to_mask] = 0
    return mask


if __name__ == "__main__":
    SEED = 1000
    torch.manual_seed(SEED)

    mask_sizes = [16, 32, 64, 80, 96]
    ps = [0, 0.25, 0.50, 0.75, 0.9, 0.95]
    # ps = [1]
    number_of_masks = 10

    for p in ps:
        for mask_size in mask_sizes:
            masks = []
            for i in range(number_of_masks):
                mask = generate_mask(p=p, mask_size=mask_size)
                masks.append(mask)

            masks = torch.stack(masks)
            print(masks.shape)
            torch.save(masks, f"data/masks_{mask_size}_{p}_{number_of_masks}.pt")
            