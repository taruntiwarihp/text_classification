* Create conda environment and install all libraries
    ```
    conda create -n bert python=3.9.16
    conda activate bert
    pip install -r requirements.txt
    # if have gpu 
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

    >>> python
        >>> import nltk
        >>> nltk.download('wordnet')
    ```

* Run process
    ```
    
    # for runing inference pipeline
    python main.py

    # for checking training logs, to check metric and loss graph
    tensorboard --logdir="logs/tensorboard_2024-11-22-14-34_bert"
    
    # finetuning script
    python train.py

    ```