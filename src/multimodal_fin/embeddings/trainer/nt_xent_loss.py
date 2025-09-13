import torch
import torch.nn.functional as F


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.5
) -> torch.Tensor:
    """
    Computes the NT-Xent (Normalized Temperature-scaled Cross Entropy) contrastive loss.

    This loss is used in self-supervised learning frameworks such as SimCLR,
    where positive pairs (z1, z2) are contrasted against all other pairs in the batch.

    Args:
        z1 (torch.Tensor): Embeddings from view 1, shape [B, D].
        z2 (torch.Tensor): Embeddings from view 2, shape [B, D].
        temperature (float): Temperature parameter for scaling logits.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    batch_size = z1.size(0)

    # Normalize embeddings to unit length
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Combine both views
    representations = torch.cat([z1, z2], dim=0)  # [2B, D]

    # Compute cosine similarity matrix: [2B, 2B]
    similarity_matrix = F.cosine_similarity(
        representations.unsqueeze(1),  # [2B, 1, D]
        representations.unsqueeze(0),  # [1, 2B, D]
        dim=2
    )

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

    # Compute positive similarities (dot product between z1 and z2)
    positives = torch.sum(z1 * z2, dim=-1)  # [B]
    positives = torch.cat([positives, positives], dim=0).unsqueeze(1)  # [2B, 1]

    # Create logits: [2B, 1 + 2B]
    logits = torch.cat([positives, similarity_matrix], dim=1)  # [2B, 1 + 2B]

    # Labels: 0 always corresponds to the positive sample
    labels = torch.zeros(2 * batch_size, dtype=torch.long).to(z1.device)

    # Compute cross-entropy loss
    loss = F.cross_entropy(logits / temperature, labels)

    return loss
