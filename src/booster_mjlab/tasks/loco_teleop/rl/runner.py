import os

import wandb
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import get_base_metadata as mjlab_get_base_metadata
from rsl_rl.utils import WandbLogWriter

from booster_mjlab.amp.runners.amp_on_policy_runner import AmpOnPolicyRunner
from booster_mjlab.rl import RslRlVecEnvWrapper
from booster_mjlab.rl.exporter_utils import attach_metadata_to_onnx

_JOINT_FIELDS = ("joint_names", "joint_stiffness", "joint_damping", "default_joint_pos")


def get_loco_teleop_metadata(env: ManagerBasedRlEnv, run_path: str) -> dict:
    """ONNX metadata where the joint fields cover only the policy's action joints.

    The joints the policy does not drive are exported as ``upper_body_*`` so the
    deploy side knows their default pose and PD gains for teleop.
    """
    md = mjlab_get_base_metadata(env, run_path)
    action_names = list(env.action_manager.get_term("joint_pos")._target_names)
    all_names = md["joint_names"]
    policy_ids = [all_names.index(n) for n in action_names]
    upper_ids = [i for i, n in enumerate(all_names) if n not in action_names]

    joint_fields = {k: md[k] for k in _JOINT_FIELDS}
    for key, values in joint_fields.items():
        md[key] = [values[i] for i in policy_ids]
        md["upper_body_" + key] = [values[i] for i in upper_ids]
    return md


class LocoTeleopAmpOnPolicyRunner(AmpOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def save(self, path: str, infos=None):
        super().save(path, infos)
        if isinstance(self.logger.writer, WandbLogWriter):
            policy_dir = os.path.dirname(path)
            filename = os.path.basename(policy_dir) + ".onnx"
            self.export_policy_to_onnx(policy_dir, filename)
            run_name = wandb.run.name if wandb.run else "local"
            onnx_path = os.path.join(policy_dir, filename)
            attach_metadata_to_onnx(
                onnx_path, get_loco_teleop_metadata(self.env.unwrapped, run_name)
            )
            if self.cfg.get("upload_model", True):
                wandb.save(onnx_path, base_path=policy_dir)
