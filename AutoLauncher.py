from MainLauncher import path_operator
from Utils import Launcher
from Utils.ConfigOperator import ConfigOperator


def main():
    class C2(ConfigOperator):
        def get_name(self, *args, **kwargs):
            return '_QueueLog'

    Launcher.Launcher(
        path_operator=path_operator,
        ##设置环境变量
        ##CUBLAS_WORKSPACE_CONFIG=:4096:8: 配置 CUDA 数学库的工作空间大小，避免某些 CUDA 操作的内存错误。
        ##LD_LIBRARY_PATH=/mnt/.../lib: 指定动态链接库的路径（如 PyTorch/CUDA 的依赖库）
        env_config='CUBLAS_WORKSPACE_CONFIG=:4096:8 LD_LIBRARY_PATH=/mnt/18t/pengxin/Softwares/Anaconda3/envs/torch1110P/torch1110/lib'
    ).launch(
        run_file='MainLauncher.py',
        ##传递配置对象，控制任务的具体行为。
        cfg=C2(cuda='0'),
        model_name='Train',
        safe_mode=False
    )


if __name__ == '__main__':
    main()
