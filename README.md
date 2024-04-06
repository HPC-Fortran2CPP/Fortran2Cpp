# F2C-Translator

## Introduction
Fortran has been a widely used programming language for scientific computation since 1957. With technological advancements, modern languages like C++ have become preferable for some projects due to their greater flexibility and features. However, the lack of an accurate and comprehensive Fortran-to-C++ translation dataset means that existing large models, including GPT-4, often struggle to perform this task effectively, resulting in translations that may fail to compile or pass unit tests. F2C-Translator aims to address this issue.

## Model
The Model is avaliable on Huggingface: [F2C-Translator](https://huggingface.co/Bin12345/F2C-Translator)

**NOTE:** We are still training the model. We will continue to update the F2C-Translator.

## Evaluation
We compared with various models (WizardCoder-15B-V1.0, CodeLlama-13b-Instruct-hf, starcoder, Magicoder-S-DS-6.7B, deepseek-coder-33b-instruct and GPT-4) on HPC_Fortran_CPP. And compared the CodeBlEU Score of the generated results.

The CodeBLEU Score Comparison is shown in the figure below:

![Example Image](Figures/CodeBLEU_Score.png)

### Reproduce Steps
1. Enter into Evaluation folder

```
cd Evaluation
```

2. Generate the results. Go the script `text_generation_pipline.py`. Add your own huggingface token to line 16. Modify the path where you want to store your results in line 55. Then select the model that you want to test between line 8 and line 13.

Run:
```
python text_generation_pipline.py
```

This will generate the results and compress each result to one line for the further CodeBLEU Score test.

3. Test CodeBLEU Score by using the following command

```
cd CodeBLEU
python calc_code_bleu.py --refs F2C-Translator/Evaluation/Groundtruth_C++.txt --hyp <path/to/your/results/txt/file> --lang cpp --params 0.25,0.25,0.25,0.25
```

## Inference and Demo
The demo code is modified from [OpenCodeInterpreter](https://github.com/OpenCodeInterpreter/OpenCodeInterpreter/tree/main/demo). Appreciate for their great project!

1. Create conda and install packages

```
cd Web_demo
conda create -n demo python=3.10
conda activate demo
pip install -r requirements.txt
```

2.  Start the demo

```
python chatbot.py
```

**NOTE:** This demo will not use the interpreter function. This feature is a potential extension for this work.

## Contact 
If you have any inquiries, please feel free to raise an issue or reach out to leib2765@gmail.com

## Citation

## Acknowledgments
