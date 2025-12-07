import os
import shutil


class ElasticRestorer:
    def __init__(self, restore_dir, checkpoint_dir=None, ckpt_prefix='last.pth.tar', local_rank=0):
        self.restore_dir = restore_dir
        self.checkpoint_dir = checkpoint_dir
        self.ckpt_prefix = ckpt_prefix
        self.local_rank = local_rank
        if self.local_rank == 0:
            self._create_restore_dir()

    def _create_restore_dir(self):
        if not os.path.exists(self.restore_dir):
            os.makedirs(self.restore_dir)

    def _force_symlink(self, src, dst):
        if os.path.exists(dst):
            os.remove(dst)
        os.link(src, dst)

    def find_resume(self):
        det_ckpt_path = os.path.join(self.restore_dir, self.ckpt_prefix)
        return det_ckpt_path if os.path.exists(det_ckpt_path) else None
    
    def save(self):
        if self.local_rank == 0:
            src_ckpt_path = os.path.join(self.checkpoint_dir, self.ckpt_prefix)
            det_ckpt_path = os.path.join(self.restore_dir, self.ckpt_prefix)
            self._force_symlink(src_ckpt_path, det_ckpt_path)
