# Fortran2Cpp

## Introduction
Fortran has been a widely used programming language for scientific computation since 1957. With technological advancements, modern languages like C++ have become preferable for some projects due to their greater flexibility and features. However, the lack of an accurate and comprehensive Fortran-to-C++ translation dataset means that existing large models, including GPT-4, often struggle to perform this task effectively, resulting in translations that may fail to compile or pass unit tests. Fortran2Cpp aims to address this issue.
Leveraging state-of-the-art Large Language Models (LLMs), Fortran2CPP employs a unique dual-agent system—the Questioner-Solver module—to iteratively translate, validate, and refine code. This process can be used to generate semantics-rich dialogue datasets that can effectively fine-tune open-weight models.


## Directory Layout

This repo has the following directory layout

```
tree -d -L 2

├── Evaluation # script to run model evaluation 
├── Figures  # some evaluation results figures
├── Web_demo # web interface demo
├── data  # sample generated datasets
├── dataset_generation  # dataset generation python programs
├── training  # model fine-tuning programs
├── training_log # some sample logs
└── utils  # other utility programs
```

## Large Language Models

You can use any open-weight models, including 
* WizardCoder-15B-V1.0,
* CodeLlama-13b-Instruct-hf,
* starcoder,
* starcoder2,
* Magicoder-S-DS-6.7B


### Steps to Use Models to Translate Fortran Codes

We recommend using virtual environment to set up the python environments and install required packages:

```
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

1. Enter into Evaluation folder

```
cd Evaluation
```

2. To generate results. Go the script `text_generation_pipline.py`. 

You can modify things like
* the model that you want to test: defined between line 9 and line 14.
* the file path where you want to store your results: defined in line 59, default is log.txt . 

Run:
```
export HUGGINGFACE_TOKEN="your_access_token_here"
python text_generation_pipline.py


# sample output of the text generation pipeline:

input_prompt: 
Translate this Fortran code to C++: 
program DRB096_doall2_taskloop_collapse_orig_no\n    use omp_lib\n    use DRB096\n    implicit none\n\n    integer :: len, i, j\n    len = 100\n\n    allocate (a(len,len))\n\n    !$omp parallel\n        !$omp single\n            !$omp taskloop collapse(2)\n            do i = 1, len\n                do j = 1, len\n                    a(i,j) = a(i,j)+1\n                end do\n            end do\n            !$omp end taskloop\n        !$omp end single\n    !$omp end parallel\n\n    print 100, a(50,50)\n    100 format ('a(50,50) =',i3)\n\nend program
#1 Fortran Code has been translated. 
Translated C++ Code:
 DRB096_doall2_taskloop_collapse_orig_no

#include <omp.h>
#include <iostream>

int main() {
    int len = 100;
    double **a = new double*[len];
    for (int i = 0; i < len; ++i) {
        a[i] = new double[len];
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop collapse(2)
            for (int i = 1; i <= len; ++i) {
                for (int j = 1; j <= len; ++j) {
                    a[i-1][j-1] += 1;
                }
            }
        }
    }

    std::cout << "a(50,50) = " << a[49][49] << std::endl;

    for (int i = 0; i < len; ++i) {
        delete[] a[i];
    }
    delete[] a;

    return 0;
}

This C++ code does the same thing as the Fortran code. It allocates a 2D array `a`, then uses OpenMP to parallelize the task of incrementing each element of `a`. The `collapse(2)` clause is used to collapse the two loops into a single loop, which is then parallelized by OpenMP. The result is printed to the console.
```

This will generate the results and compress each result to one line for the further CodeBLEU Score test.

This script does the following
* model: Bin12345/F2C-Translator // configurable to use any other models
* dataset: Bin12345/HPC_Fortran_CPP
* Translate Fortran Code in the dataset to C++ code
* Write to a log

3. Test CodeBLEU Score by using the following command

```
cd CodeBLEU
python calc_code_bleu.py --refs Fortran2Cpp/Evaluation/Groundtruth_C++.txt --hyp <path/to/your/results/txt/file> --lang cpp --params 0.25,0.25,0.25,0.25
```

* Run inference on a Slurm Cluster: Your should use this script to start the inference: `sbatch <The/following/script>`

```
#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu&hbm80g
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -J model_training
#SBATCH --mail-user=<Your/Email>
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00
#SBATCH -A <Your/Project>

# Load conda  
echo "loading conda..."
module load conda 
conda activate <Your/Conda/env>

# Huggingface Setting 
echo "Setting Huggingface..."
export HF_HOME=$SCRATCH/huggingface 
export HF_TOKEN=<Your/HF/ToKen>

# OpenMP settings:
echo "Setting OMP..."
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Set CFLAGS and LDFLAGS and CUTLASS 
export CFLAGS="-I/<Your/Conda/env>/include $CFLAGS"
export LDFLAGS="-L/<Your/Conda/env>/lib $LDFLAGS"
export CUTLASS_PATH=$HOME/cutlass

# run the application:
echo "Start to run the inference..."
chmod +x <Your/inference/file/path>
srun -n 1 -c 8 --cpu_bind=cores -G 4 --gpu-bind=none  <Your/inference/file/path> > <Your/log/file/path> 2>&1 
```

## Dataset Generation 

The dialogue dataset in **Fortran2CPP** is generated using a structured, automated conversation between two specialized AI agents called the **Questioner** and the **Solver**. Here's how it works, in simple terms:

1. **Initial Translation**:
   - The **Questioner** first presents the original Fortran code and asks the **Solver** to translate it into C++.
   - The **Solver** responds by creating the initial translated version of the C++ code.

2. **Unit Test Generation**:
   - The **Questioner** then requests unit tests—small pieces of code used to automatically check if the translation is correct—for both the original Fortran code and the new C++ version.
   - The **Solver** generates these tests, ensuring that both the Fortran and C++ versions behave identically.

3. **Compilation and Execution**:
   - Both Fortran and C++ code are automatically compiled (converted into runnable programs) and executed.
   - If either program fails at this step (due to errors or bugs), the errors are noted.

4. **Iterative Refinement**:
   - Using feedback from compilation and execution, the **Questioner** points out the errors and asks the **Solver** to correct the C++ translation.
   - The **Solver** corrects the code, and the process repeats until the errors are resolved or a maximum number of iterations is reached.

5. **Final Verification**:
   - After iterative improvements, the **Questioner** checks if the outputs of the Fortran and C++ codes match exactly, confirming the translation is correct.

Throughout these interactions, the back-and-forth conversations—questions, answers, errors, and solutions—are systematically recorded, forming the **multi-turn dialogue dataset**. Each dialogue captures a real translation scenario, including code problems and detailed error-fixing steps, making it highly effective for training artificial intelligence models to better translate and understand programming languages.

1. Setup your OpenAI Key.
```
cd dataset_generation
export OPENAI_API_KEY="sk...."
```

Modify engine_F2C.py to customize the input dataset range, teacher model ID, output folder, etc.
Default values are the following:
```
if __name__ == "__main__":
    key = get_openai_api_key() # Obtain your OpenAI Key from an environment variable named OPENAI_API_KEY

    # You can also use other datasets
    Fortran_dataset = load_dataset("codeparrot/github-code", "FORTRAN-all")
    data = Fortran_dataset["train"][78000:85000]

    output_file = "" # Output Json file
    generate_data(key, data, output_file, gpt_model="gpt-4o") # Need to change
```

2. Start the dataset generation

```
python engine_F2C.py
```

A sample log file and json file are stored in 
* dataset_generation/example_results

The dataset that we used is included in `F2C-Translator/data/F2C_dialogue_2.5K.json` file.

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

## Fine-tuning Models using the Dialogue Dataset

1. Install packages
```
deepspeed==0.12.3
pyarrow==13.0.0
torch==2.0.1
numpy==1.26.4
```

2. Collect the data
If you only want to use the code pairs.
Put them into a two column csv file should be fine.

If you would like to fine-tune the multi-dialogue dataset.
handle data like this:
```
<User1><Assistant1><User2><Assistant2><User3><Assistant3>

->

1. <User1>                                            <Assistant1>
2. <User1><Assistant1><User2>                         <Assistant2>
3. <User1><Assistant1><User2><Assistant2><User3>      <Assistant3>
```
Put the first part of message to the first colunm of csv file and the last part(`<Assistant1,2,3>`) to the second column.

You can achieve that by runing this command `python training/utils/data/splitting_multiturns_dialogue.py`

Input file is `data/F2C_dialogue_2.5K.json`

The converted json file is shown in `data/F2C_dialogue_2.5K_test.json`

For example, input Json

```json
[
    {
        "id": "conv1",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good, thank you."}
        ]
    }
]
```
will be split into the following output json

```json
[
    {
        "id": "conv1",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"}
        ]
    },
    {
        "id": "conv1",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good, thank you."}
        ]
    }
]
```


3. Start traning by running

```
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR="localhost"
export MASTER_PORT="12355"
export LOCAL_RANK=8 # GPU number
python NLP_task_training.py --model_name_or_path Qwen/Qwen2.5-72B-Instruct
```

**NOTE:** This demo will not use the interpreter function. This feature is a potential extension for this work.

## Hardware requirements

We used 6 A100 GPUs with 80GB memory for the training.  (Use Lora)

We used 2 A100 GPUs with 80GB memory for the inference. 

## Contact 
If you have any inquiries, please feel free to raise an issue or reach out to leib2765@gmail.com.

## Citation

Fortran2CPP: Automating Fortran-to-C++ Translation using LLMs via Multi-Turn Dialogue and Dual-Agent Integration, Dec. 2024, https://arxiv.org/abs/2412.19770 

## Known Issues

https://huggingface.co/datasets/Bin12345/HPC_Fortran_CPP seems to have 315 rows of data. However, Groundtruth_C++.txt has only 296 rows.  The reason is that we filtered some long data samples. 

