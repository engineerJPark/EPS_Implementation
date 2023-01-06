# Implementation for EPS(Explicit Pseudo-pixel Supervision)

This is the reproduction for Explicit Pseudo-pixel Supervision (EPS), which is accepted in CVPR 2021.

Paper Link : [here](https://openaccess.thecvf.com/content/CVPR2021/html/Lee_Railroad_Is_Not_a_Train_Saliency_As_Pseudo-Pixel_Supervision_for_CVPR_2021_paper.html)

EPS is method for Weakly Supervised Semantic Segmentation, and they use localization map and saliency map together for improving pseudo-label quality.

Implementation detail might differ from the [original implementation.](https://github.com/halbielee/EPS)

Do this for 
```
cd eps && bash do.sh
```

# Dependency & TroubleShooting


for installing overall depency of this repo, follow this installation : 

```
sudo apt update
sudo apt install build-essential
pip install matplotlib
pip install imageio

sudo apt-get remove cython
pip install -U cython
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

for spend some lazy times... use tmux.

```
sudo apt-get install tmux
```

# Acknowledgements

Base code is from J.Ahn's [repository](https://github.com/jiwoon-ahn/irn). Thanks for original repository writer. Applying denseCRF and AffinityNet(part of IRN) is heavily done from original repository.
