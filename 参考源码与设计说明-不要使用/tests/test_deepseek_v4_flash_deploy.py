"""Guard expensive model startup when the rented GPU memory is insufficient."""
import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

from deepseek_v4_flash.deepseek_v4_flash import build_command, check_hardware


class TestDeepSeekV4FlashDeploy(unittest.TestCase):
    def _args(self, count):
        return Namespace(
            variant="preview", model=None, tensor_parallel_size=count,
            max_model_len=16384, gpu_memory_utilization=0.9,
            host="127.0.0.1", port=8000, served_model_name="deepseek-v4-flash",
        )

    @patch("torch.cuda.get_device_name", return_value="A100 80GB")
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.device_count")
    @patch("torch.cuda.is_available", return_value=True)
    def test_rejects_one_80gb_and_accepts_four(self, _available, count, props, _name):
        props.return_value = SimpleNamespace(total_memory=80_000_000_000)
        count.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "170GB"):
            check_hardware(self._args(1))
        count.return_value = 4
        self.assertIn("A100 80GB", check_hardware(self._args(4)))

    def test_command_keeps_variant_and_context_explicit(self):
        args = self._args(4)
        command = build_command(args)
        self.assertEqual(command[:3], ["vllm", "serve", "deepseek-ai/DeepSeek-V4-Flash"])
        self.assertEqual(command[command.index("--max-model-len") + 1], "16384")
        args.variant = "0731"
        self.assertEqual(build_command(args)[2], "deepseek-ai/DeepSeek-V4-Flash-0731")


if __name__ == "__main__":
    unittest.main()
