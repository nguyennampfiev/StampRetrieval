## Source code implementation of Grounding DINO and MGPSA for document retrieval.
### Pipeline
 - ` 1. Stamp extraction with finetuning [Grounding DINO](https://arxiv.org/abs/2303.05499)`
 - ` 2. Finetuning of [MPSA](https://ieeexplore.ieee.org/document/10638479) for stamp feature extraction.`
 - ` 3. Stamp retrieval with KNNs.`
### To run demo with Steamlit, just build docker image from source. Make sure you already installed docker on your computer
```
docker build -t retrieval-stamp .
```
then to see demo
```
docker run --rm -it -p 8501:8501 retrieval-stamp
```

### To run demo with Gradio,
```
python app_gradio.py
```

