import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from nerfview import CameraState

from flow3d.scene_model import SceneModel
from flow3d.vis.utils import draw_tracks_2d_th, get_server
from flow3d.vis.viewer import DynamicViewer


class Renderer:
    def __init__(
        self,
        model: SceneModel,
        device: torch.device,
        # Logging.
        work_dir: str,
        port: int | None = None,
    ):
        self.device = device

        self.model = model
        self.num_frames = model.num_frames

        self.work_dir = work_dir
        self.global_step = 0
        self.epoch = 0

        self.viewer = None
        if port is not None:
            server = get_server(port=port)
            self.viewer = DynamicViewer(
                server, self.render_fn, model.num_frames, work_dir, mode="rendering"
            )

        self.tracks_3d = self.model.compute_poses_fg(
            #  torch.arange(max(0, t - 20), max(1, t), device=self.device),
            torch.arange(self.num_frames, device=self.device),
            inds=torch.arange(10, device=self.device),
        )[0]

    @staticmethod
    def init_from_checkpoint(
        path: str, device: torch.device, *args, **kwargs
    ) -> "Renderer":
        guru.info(f"Loading checkpoint from {path}")
        ckpt = torch.load(path)
        state_dict = ckpt["model"]
        model = SceneModel.init_from_state_dict(state_dict)
        model = model.to(device)
        renderer = Renderer(model, device, *args, **kwargs)
        renderer.global_step = ckpt.get("global_step", 0)
        renderer.epoch = ckpt.get("epoch", 0)
        return renderer

    @torch.inference_mode()
    def render_fn(self, camera_state: CameraState, img_wh: tuple[int, int]):
        if self.viewer is None:
            return np.full((img_wh[1], img_wh[0], 3), 255, dtype=np.uint8)

        W, H = img_wh

        focal = 0.5 * H / np.tan(0.5 * camera_state.fov).item()
        K = torch.tensor(
            [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
            device=self.device,
        )
        w2c = torch.linalg.inv(
            torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
        )
        t = (
            int(self.viewer._playback_guis[0].value)
            if not self.viewer._canonical_checkbox.value
            else None
        )
        self.model.training = False
        img = self.model.render(t, w2c[None], K[None], img_wh)["img"][0]
        if not self.viewer._render_track_checkbox.value:
            img = (img.cpu().numpy() * 255.0).astype(np.uint8)
        else:
            assert t is not None
            tracks_3d = self.tracks_3d[:, max(0, t - 20) : max(1, t)]
            tracks_2d = torch.einsum(
                "ij,jk,nbk->nbi", K, w2c[:3], F.pad(tracks_3d, (0, 1), value=1.0)
            )
            tracks_2d = tracks_2d[..., :2] / tracks_2d[..., 2:]
            img = draw_tracks_2d_th(img, tracks_2d)
        return img
