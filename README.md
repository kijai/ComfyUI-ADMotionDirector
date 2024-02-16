# ComfyUI nodes to use AnimateDiff-MotionDirector

https://github.com/kijai/ComfyUI-ADMotionDirector/assets/40791699/b082cbb0-4ba8-4ee6-9a2a-48123dca41ef

![admd_example_workflow](https://github.com/kijai/ComfyUI-ADMotionDirector/assets/40791699/6d7b6ae4-da1c-4afd-a72c-a7642363c69a)

Install 'pip install -r requirements.txt'

For portable: 'python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-ADMotionDirector\requirements.txt'

Tested with pytorch 2.1.1 + cu121 and 2.2.0 + cu121, older ones may have issues. 

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom nodes for using [AnimateDiff-MotionDirector](https://github.com/ExponentialML/AnimateDiff-MotionDirector)

After training, the LoRAs are intended to be used with the ComfyUI Extension [ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved).


## BibTeX

```
@article{guo2023animatediff,
  title={AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning},
  author={Guo, Yuwei and Yang, Ceyuan and Rao, Anyi and Wang, Yaohui and Qiao, Yu and Lin, Dahua and Dai, Bo},
  journal={arXiv preprint arXiv:2307.04725},
  year={2023}
}
@article{zhao2023motiondirector,
  title={MotionDirector: Motion Customization of Text-to-Video Diffusion Models},
  author={Zhao, Rui and Gu, Yuchao and Wu, Jay Zhangjie and Zhang, David Junhao and Liu, Jiawei and Wu, Weijia and Keppo, Jussi and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2310.08465},
  year={2023}
}
```

## Disclaimer
This project is released for academic use and creative usage. We disclaim responsibility for user-generated content. Users are solely liable for their actions. The project contributors are not legally affiliated with, nor accountable for, users' behaviors. Use the generative model responsibly, adhering to ethical and legal standards.

## Acknowledgements
Codebase built upon:
- [AnimateDiff-MotionDirector](https://github.com/ExponentialML/AnimateDiff-MotionDirector)
- [AnimateDiff](https://github.com/guoyww/AnimateDiff)
- [Tune-a-Video](https://github.com/showlab/Tune-A-Video).
- [MotionDirector](https://github.com/showlab/MotionDirector)
- [Text-To-Video-Finetuning](https://github.com/ExponentialML/Text-To-Video-Finetuning)
- [lora](https://github.com/cloneofsimo/lora)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

