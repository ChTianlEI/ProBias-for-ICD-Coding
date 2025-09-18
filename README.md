# ProBias: Directed Bipartite Graph Encoder with Probability-based Bias for Multi-label ICD Coding

This repository provides the implementation of our ProBias model for the automatic ICD coding task. The experiments are conducted on the full versions of three benchmark datas: `MIMIC-III-ICD-9`, `MIMIC-IV-ICD-9`, and `MIMIC-IV-ICD-10`.

## Environment
Install the required environment using the `requirements.txt` file.

## Dataset
1. **Obtain Licenses**: First, you need to obtain licenses to download the [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) and [MIMIC-IV](https://physionet.org/content/mimic-iv-note/2.2/) datasets.
2. **Data Processing**:
    - Process the MIMIC-III dataset following the steps in [caml-mimic](https://github.com/jamesmullenbach/caml-mimic).
    - Process the MIMIC-IV dataset according to [MIMIC-IV-ICD](https://github.com/thomasnguyen92/MIMIC-IV-ICD-data-processing).
3. **Format Conversion**: Follow [ICD-MSMN](https://github.com/GanjinZero/ICD-MSMN/tree/master) to obtain the JSON - formatted files required for our model.

## Preparation for Model Training
1. **Generate Datasets**: Run `/preprocess/labelizer.ipynb` to generate the train, validation, and test datasets. This will create text files and one - hot label files in the format `{dataset}_{name}.pkl` and `{dataset}_{name}_1hot.npz`.
2. **Create Guidance Graph Files**: Run `/preprocess/co-occurrence_encoding_utils.ipynb` to generate the guidance graph - related files:
    - `{dataset}_prob_matrix.npy`: Conditional co - occurrence probability matrix.
    - `{dataset}_adj_matrix.pkl`: Binary co - occurrence matrix between rare and common codes.
    - `{dataset}_ground_ind_rare.pkl`: Indices of rare codes in the lexicographically sorted codes set.
    - `{dataset}_ground_ind_common.pkl`: Indices of common codes in the lexicographically sorted codes set.
    - `{dataset}_c_indices.pkl`: Binning indices of the co-occurrence encoding item in the guidance graph.
3. **Enhance Code Descriptions**: Run `/preprocess/generate_code_des.ipynb` to generate enhanced code descriptions using LLMs, saved as `enhanced_icd_{dataset}.json`.
4. **Tokenize Code Descriptions**: Run `/preprocess/label_tokenizer.ipynb` to obtain the input tokens of the code descriptions for the guidance graph, saved as `icd_{dataset}_desc.pkl`.

**Note**: `dataset` can be `mimic3`, `mimic4_icd9`, or `mimic_4_icd10`; `name` can be `train`, `val`, or `test`.

## Project Directory Structure
After data preprocessing, the generated files will be stored in the `model_data` directory. The structure of the directory is illustrated in `Directory Structure.txt`

The `model_core` directory contains:
- A `models` directory:
    - `graph.py`: Encodes code guidance correlations.
    - `model.py`: Defines other model structures.
- A `model_support` directory:
    - `dataset.py`: Processes the train, validation, and test datasets for input.
    - `eval_metrics.py`: Contains evaluation metrics.
    - `trainer.py`: Contains training helper functions.

- `config.py`: Used to modify corresponding hyperparameters, including:
    - `TRAINING ARGUMENTS`: You can set epochs, gradient accumulation steps, seeds, learning rate, etc.
    - `EXPERIMENT SETTINGS`: You can set hidden size, number of heads, dataset, and graph - related parameters.
- `main.py`: The main script to run the ProBias model.

## GPU Memory Requirements
- Experiments on the MIMIC-III-ICD-9 and MIMIC-IV-ICD-9 datasets require at least a single GPU with 48GB of memory.
- Experiments on the MIMIC-IV-ICD-10 dataset should be run on a single GPU with 80GB of memory.

## Running the Model
### Training
1. Open `model_core/config.py` and set `MODE = "train"`.
2. Run `python main.py`.

### Testing
1. Open `model_core/config.py` and set `MODE = "test"` and `START_MODEL_FROM_CHECKPOINT = {checkpoint_path}`.
2. Run `python main.py`.