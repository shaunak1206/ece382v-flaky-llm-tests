import pytest
import torch
from torch_embed_sim import EmbeddingSim


def test_embedding_sim_init_default():
    sim = EmbeddingSim()
    assert sim.similarity_measure in {"cosine", "dot", "l2"}
    assert isinstance(sim.normalize, bool)


@pytest.mark.parametrize("measure", ["cosine", "dot", "l2"])
def test_embedding_sim_init_valid_measures(measure):
    sim = EmbeddingSim(similarity_measure=measure)
    assert sim.similarity_measure == measure


@pytest.mark.parametrize("measure", ["invalid", "", None, 123])
def test_embedding_sim_init_invalid_measure_raises(measure):
    with pytest.raises(ValueError):
        EmbeddingSim(similarity_measure=measure)


def test_embedding_sim_call_with_2d_tensors():
    sim = EmbeddingSim(similarity_measure="cosine")
    emb1 = torch.randn(10, 128)
    emb2 = torch.randn(10, 128)
    scores = sim(emb1, emb2)
    assert scores.shape == (10,)
    assert scores.dtype == emb1.dtype


def test_embedding_sim_call_with_1d_tensors():
    sim = EmbeddingSim(similarity_measure="dot")
    emb1 = torch.randn(128)
    emb2 = torch.randn(128)
    score = sim(emb1, emb2)
    assert isinstance(score, torch.Tensor)
    assert score.shape == ()
    assert score.dtype == emb1.dtype


@pytest.mark.parametrize("measure", ["cosine", "dot", "l2"])
def test_embedding_sim_call_batch_consistency(measure):
    sim = EmbeddingSim(similarity_measure=measure)
    emb1 = torch.randn(5, 64)
    emb2 = torch.randn(5, 64)
    batch_scores = sim(emb1, emb2)
    assert batch_scores.shape == (5,)

    # Check individual scores match batch score
    for i in range(5):
        s = sim(emb1[i], emb2[i])
        if measure == "l2":
            assert pytest.approx(float(batch_scores[i].item()), rel=1e-5) == float(s.item())
        else:
            # Dot and cosine may be negative so allow normal float comparison
            assert abs(batch_scores[i].item() - s.item()) < 1e-4


def test_embedding_sim_call_shape_mismatch_raises():
    sim = EmbeddingSim()
    emb1 = torch.randn(10, 16)
    emb2 = torch.randn(12, 16)
    with pytest.raises(RuntimeError):
        sim(emb1, emb2)


def test_embedding_sim_normalize_true_cosine():
    sim = EmbeddingSim(similarity_measure="cosine", normalize=True)
    emb1 = torch.randn(3, 5)
    emb2 = torch.randn(3, 5)
    scores = sim(emb1, emb2)
    assert torch.all((scores >= -1) & (scores <= 1))


def test_embedding_sim_normalize_true_dot():
    sim = EmbeddingSim(similarity_measure="dot", normalize=True)
    emb1 = torch.randn(3, 5)
    emb2 = torch.randn(3, 5)
    scores = sim(emb1, emb2)
    # Dot with normalize=True should behave like cosine similarity roughly
    assert torch.all((scores >= -1) & (scores <= 1))


def test_embedding_sim_l2_similarity_values():
    sim = EmbeddingSim(similarity_measure="l2", normalize=False)
    emb1 = torch.tensor([[1., 2.], [3., 4.]])
    emb2 = torch.tensor([[1., 2.], [0., 0.]])
    scores = sim(emb1, emb2)
    # L2 similarity should be -||x-y|| where ||.|| is Euclidean distance
    expected = -torch.norm(emb1 - emb2, dim=1)
    assert torch.allclose(scores, expected)


def test_embedding_sim_call_different_dims_raises():
    sim = EmbeddingSim()
    emb1 = torch.randn(4, 10)
    emb2 = torch.randn(4, 11)
    with pytest.raises(RuntimeError):
        sim(emb1, emb2)


def test_embedding_sim_call_invalid_input_dims_raises():
    sim = EmbeddingSim()
    emb1 = torch.randn(3, 4, 5)
    emb2 = torch.randn(3, 4, 5)
    with pytest.raises(RuntimeError):
        sim(emb1, emb2)


def test_embedding_sim_call_single_vector_mismatched_dims_raises():
    sim = EmbeddingSim()
    emb1 = torch.randn(128)
    emb2 = torch.randn(64)
    with pytest.raises(RuntimeError):
        sim(emb1, emb2)


def test_embedding_sim_call_empty_batch_raises():
    sim = EmbeddingSim()
    emb1 = torch.empty((0, 10))
    emb2 = torch.empty((0, 10))
    with pytest.raises(RuntimeError):
        sim(emb1, emb2)


def test_embedding_sim_call_on_cpu_and_gpu_similar_outputs():
    if torch.cuda.is_available():
        sim_cpu = EmbeddingSim()
        sim_gpu = EmbeddingSim()
        emb1 = torch.randn(4, 8)
        emb2 = torch.randn(4, 8)
        out_cpu = sim_cpu(emb1, emb2)
        out_gpu = sim_gpu(emb1.cuda(), emb2.cuda()).cpu()
        assert torch.allclose(out_cpu, out_gpu, rtol=1e-4, atol=1e-6)


def test_embedding_sim_call_with_normalize_false_cosine_and_dot():
    emb1 = torch.randn(3, 4)
    emb2 = torch.randn(3, 4)

    sim_cos = EmbeddingSim(similarity_measure="cosine", normalize=False)
    out_cos = sim_cos(emb1, emb2)
    assert out_cos.shape == (3,)

    sim_dot = EmbeddingSim(similarity_measure="dot", normalize=False)
    out_dot = sim_dot(emb1, emb2)
    assert out_dot.shape == (3,)
    # Dot product values can be any real number
    assert (out_dot > float('-inf')).all()


def test_embedding_sim_call_normalize_affects_dot_and_cosine():
    emb1 = torch.randn(4, 6)
    emb2 = torch.randn(4, 6)

    sim1 = EmbeddingSim(similarity_measure="dot", normalize=False)
    sim2 = EmbeddingSim(similarity_measure="dot", normalize=True)
    out1 = sim1(emb1, emb2)
    out2 = sim2(emb1, emb2)
    # Normalized dot is roughly cosine and should have smaller magnitude
    assert torch.mean(out2.abs()) <= torch.mean(out1.abs())

    sim3 = EmbeddingSim(similarity_measure="cosine", normalize=False)
    sim4 = EmbeddingSim(similarity_measure="cosine", normalize=True)
    out3 = sim3(emb1, emb2)
    out4 = sim4(emb1, emb2)
    # Cosine similarity must be in [-1,1] when normalized
    assert torch.all((out4 >= -1) & (out4 <= 1))


def test_embedding_sim_call_different_batch_and_vec_dim_raises():
    sim = EmbeddingSim()
    emb1 = torch.randn(3, 5)
    emb2 = torch.randn(5)
    with pytest.raises(RuntimeError):
        sim(emb1, emb2)


def test_embedding_sim_call_single_vector_and_2d_batch_raises():
    sim = EmbeddingSim()
    emb1 = torch.randn(5)
    emb2 = torch.randn(3, 5)
    with pytest.raises(RuntimeError):
        sim(emb1, emb2)
