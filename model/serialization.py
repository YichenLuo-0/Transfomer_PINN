import torch


def encode_z(x, y, depth=16):
    key = torch.zeros_like(x, dtype=torch.int64)
    for i in range(depth):
        # get the i-th bit of x and y
        x_bit = (x // (2 ** i)) % 2
        y_bit = (y // (2 ** i)) % 2
        # combine the new key value based on x_bit and y_bit
        key += (x_bit * (2 ** (2 * i + 0))) + (y_bit * (2 ** (2 * i + 1)))
    return key


def normalize(x, depth=16):
    # The input x is of shape [..., sequence, 1], where the last dimension is the coordinate
    sequence_dim = x.dim() - 2

    # Find the min and max value of x
    min_val = torch.min(x, dim=sequence_dim).values
    max_val = torch.max(x, dim=sequence_dim).values

    # Expand the min_val and max_val to the same shape as x
    max_val = max_val.unsqueeze(-1).expand(*max_val.shape[:-1], x.shape[sequence_dim], max_val.shape[-1])
    min_val = min_val.unsqueeze(-1).expand(*min_val.shape[:-1], x.shape[sequence_dim], min_val.shape[-1])

    # Transform the coordinates to [0, 2^depth-1]
    normalized = (x - min_val) / (max_val - min_val) * (2 ** depth - 1)
    return normalized.to(torch.int64)


def serialization(coords, bc, depth=16):
    x, y = coords[..., 0:1], coords[..., 1:2]
    batch_size, _, d_bc = bc.shape

    # Convert x, y coordinates to integers
    xx = normalize(x, depth=depth)
    yy = normalize(y, depth=depth)

    # Code the coordinates using Z curve or Hilbert curve
    code = encode_z(xx, yy, depth=depth).reshape(batch_size, -1)

    # Sort the input tensor based on the code
    _, indices = torch.sort(code, dim=1)
    return indices, code.unsqueeze(-1)


def sort_tensor(x, indices):
    _, _, d_out = x.shape
    indices_out = indices.unsqueeze(-1).expand(*indices.shape, d_out)

    # Resort the output tensor based on the indices
    return torch.gather(x, 1, indices_out)


def desort_tensor(x, indices):
    _, _, d_out = x.shape
    indices_out = indices.unsqueeze(-1).expand(*indices.shape, d_out)

    # Resort the output tensor based on the indices
    return x.scatter(1, indices_out, x)
