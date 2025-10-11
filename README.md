# Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization  (NeurIPS 2025 Poster)

This is the source code for the LTW watermark algorithms presented in the paper.

## In summary, our contributions are as follows:

• Analysis: We find existing selective watermarking method underexplored potentially informative factors that may be used as criterions for selection. We are the first to propose the method of utilizing a trained network to make decisions on whether to selectively apply watermark, unveiling a new perspective of selective watermarking strategies.

• Method: We propose LTW, a novel selective watermarking framework that uses a trained lightweight network for selectively watermarking LLMs. We introduce LTW-1 and LTW-0, by applying our selective framework to baseline watermark KGW and Unigram.

• Evaluation: We conducted extensive experiments across multiple models, demonstrating the high text quality and detectability of our methods. We surpass previous watermarking methods in text quality, having the least perplexity while without compromising detectability.



## Quickstart:

As a quickstart, you can find the plots and code for the tables in our experiments in the eval folder.

```
# you can try our method using the generate.ipynb
# You can train a Selective watermarking model with the opt model using train.py
python train.py
```

## Reference BibTeX

```
@inproceedings{chenrui25neurips,
  title={Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization},
    author = {Chenrui Wang and Junyi Shu and Billy Chiu  and Yu Li and Saleh Alharbi and Min Zhang and Jing Li},
  booktitle = {The Thirty-Ninth Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year={2025}
}


```

