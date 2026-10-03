"""CPU regression checks for binary training and evaluation contracts."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from training import build_loss_fn
from models import build_model
from eval.idfraud.metrics import count_tp_tn_fp_fn


class BinaryClassificationTests(unittest.TestCase):
    def test_focal_probability_and_logit_losses_and_gradients_agree(self):
        logits = torch.tensor([-4., -1., 0., 1., 4.], requires_grad=True)
        labels = torch.tensor([0., 1., 0., 1., 0.])
        params = dict(alpha=0.65, gamma=3.)
        probability_loss = build_loss_fn('binary_focal', **params)(logits.sigmoid(), labels)
        logit_loss = build_loss_fn('binary_focal', from_logits=True, **params)(logits, labels)
        torch.testing.assert_close(probability_loss, logit_loss)
        probability_grad = torch.autograd.grad(probability_loss, logits, retain_graph=True)[0]
        logit_grad = torch.autograd.grad(logit_loss, logits)[0]
        torch.testing.assert_close(probability_grad, logit_grad)
        self.assertTrue(torch.isfinite(logit_grad).all())

    def test_all_classifier_heads_support_focal_backward(self):
        for head in ('legacy_v1', 'legacy_v2', 'v1', 'legacy'):
            with self.subTest(head=head), patch('models.load_backbone', return_value=(nn.Linear(8, 8), 8)):
                model = build_model('test', 'cpu', head_type=head, pretrained=False)
                output = model(torch.randn(4, 8))
                self.assertEqual(output.shape, (4, 1))
                loss = build_loss_fn('binary_focal', from_logits=head in ('v1', 'legacy'))(
                    output, torch.tensor([[0.], [1.], [0.], [1.]])
                )
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_metrics_preserve_probabilities_and_convert_logits(self):
        probabilities = torch.tensor([0.2, 0.8, 0.4, 0.9])
        labels = torch.tensor([0, 1, 1, 0])
        for output_type, values in [('probs', probabilities), ('logits', probabilities.logit())]:
            with self.subTest(output_type=output_type):
                self.assertEqual(count_tp_tn_fp_fn(values, labels, output_type), (1, 1, 1, 1))


if __name__ == '__main__':
    unittest.main()
