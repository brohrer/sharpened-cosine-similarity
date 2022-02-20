import torch

from sharpened_cosine_similarity import SharpenedCosineSimilarity, SharpenedCosineSimilarity_ConvImpl

def test():
    original = SharpenedCosineSimilarity(5, 5, 3)
    faster = SharpenedCosineSimilarity_ConvImpl(5, 5, 3)
    faster.load_state_dict(original.state_dict())

    test_values = torch.randn(1, 5, 32, 32)

    orig_output = original(test_values)
    faster_output = faster(test_values)

    print((orig_output - faster_output).abs().max().item())

test()
