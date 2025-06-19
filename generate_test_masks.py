import torch
import pickle

def generate_mask(p, mask_size):
    assert 0 <= p <= 1
    random_mask = torch.rand(mask_size, mask_size) < p
    return random_mask


if __name__ == "__main__":
    SEED = 1000
    torch.manual_seed(SEED)

    mask_sizes = [16, 32, 64, 80, 96]
    ps = [0.50, 0.75, 0.95]
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
            