# Cross-lingual Multi-Level Adversarial Transfer to Enhance Low-Resource Name Tagging 

This repository includes the source code for the cross-lingual name tagging with multi-level adversarial training


## Model
 
   ![Alt Text](https://drive.google.com/drive/u/0/folders/19H9trOJjTBDDLyUux76WP77K8qaCHlrF)

## Requirements

Python3, Pytorch

## Data Format

* Label format

    The name tagger follows *BIO* or *BIOES* scheme:
    
    ![Alt Text](https://blender04.cs.rpi.edu/~zhangb8/public_misc/bio_scheme_example.png)

* Sentence format
    
    Document is segmented into sentences. Each sentence is tokenized into multiple tokens. 
    
    In the training file, sentences are separated by an empty line. Tokens are separated by linebreak. For each token, label should be always at the end. Token and label are separated by space.
        
    Example:
    ```
    George B-PER
    W. I-PER
    Bush I-PER
    went O
    to O
    Germany B-GPE
    yesterday O
    . O
    
    New B-ORG
    York I-ORG
    Times I-ORG
    ```
    
    A real example of a bio file: `example/data/eng.train.bio`
        

## Usage

Training example is provided in `example/seq_labeling_naacl/`.

## Citation

[1] Lifu Huang, Heng Ji, Jonathan May. [Cross-lingual Multi-Level Adversarial Transfer to Enhance Low-Resource Name Tagging](http://nlp.cs.rpi.edu/paper/adversarial2019.pdf), Proc. NAACL, 2019
