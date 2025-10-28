Project Proposal (Weightage 25% & Due Date: Week 3)
A list of suggested projects and data sets are provided at the links above. Read the list carefully
and decide a precise problem setting and dataset for your proposal. We would discourage changing
the core datasets altogether later during the project. Page limit: Proposals should be one page
maximum. Include the following information:
• Project title
• Data set to be used.
• Project idea. This should be approximately one-two paragraphs.
• Software you are planning to develop (mention core libraries you are planning to use).
• Papers to read. Include 1-3 relevant papers. It will be best to carefully read at least one of
them before submitting your proposal.


--- 
Quantization LLM

- Use: https://github.com/vllm-project/llm-compressor/tree/main
Methods: GPTQ, AWQ

Weight: 4bit, 8bit
Activation: 8bit, 16bit

Calibration Dataset: gsm-8k, LAMBADA, ultrachat_200k
Evaluation Dataset: gsm-8k, LAMBADA.


# 0.5B 
## GSM8k
baseline: 34.8
RTN: 5.6
Calibration: GPTQ - w4a16
    512, Acc: 24
    1024, Acc: 24.4
    2048, Acc: 29.6
    4096, Acc: 25.2 
AWQ: 2048, Acc: 16.8 

## wikitext - ultrachat
baseline: 18.11
RTN: 24.57
GPTQ 2048: 20.69
AWQ 2048: 21.58

## lambada
baseline: accuracy: 52.4, perplexity: 12.57
RTN: acc: 34, perflexity: 26.96
GPTQ: acc 44.4, perplexity: 19.23
AWQ: acc: 46.4, perplexity: 16.89

## mixed dataset
w4a16, w4a4, w8a8 fp, w8a8 int -> not priority 


## Efficiency Metrics
Storage
Throughput: Tokens per second or samples per second.
Time-to-first token
Peak GPU RAM