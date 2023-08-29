This project has 3 parts:
1. light estimation and probe prediction
2. relighting 
3. generate hdr output predictions from ldr input images

Sor for each of this individual tasks, there are individual folders as well, which contains the corresponding models and evaluation scripts.

Datasets can be downloaded from the following link:

https://drive.google.com/drive/folders/15Bhw12W7Jes8d9vHZWIPDtzQAVvZrvAm



* To see the output of over all tasks, run the "lightEstim.py" file in cell. 
* To see the ouput of evaluation task, run the file "probepred_eval.sh" or "relight_eval.sh" (Runs on PyTorch 1.3.1 and CUDA 11.00)

Example script of "probepred_eval.sh": 

set -e
export PYTHONPATH=.

scene=bright

mkdir -p eval_output/probe_predict
for light_dir in `seq 6 7`; do

python3 -m probe_predict.ProbeEval \
--light_dir ${light_dir} \
--out eval_output/probe_predict/${scene}_dir${light_dir}.jpg \
${scene}
done



* HDR images from LDR input can be found by following these steps: ( It is recommended to use the PyTorch library along with OpenCV. Python3.6 or higher)

1. The hdr.py script takes an LDR image as input, and generates an HDR Image prediction, (.hdr or .exr file if --use_exr flag is used).

run the script as follow:

python hdr.py  ldr_input.jpg (takes 1 input)

2. In order to take the whole directory as input:

python hdr.py  path/to/ldr_dir

3. To save results in a separate directory:

python hdr.py  *.jpg --out results/


The AR application can be found from ARdeepLight/build folder. 
