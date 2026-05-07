import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from snapkv_utils import SnapKVCluster
import json

MODEL_PATH = "/workspace/models/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1"

# 测试数据：长文档 + 问题
TEST_CASES = [
    {
        "context": """The transformer architecture was introduced in the paper "Attention is All You Need" 
        by Vaswani et al. in 2017. It relies entirely on attention mechanisms, dispensing with recurrence 
        and convolutions entirely. The model consists of an encoder and decoder, each composed of 
        multiple layers. Each layer has two sub-layers: a multi-head self-attention mechanism and a 
        position-wise fully connected feed-forward network. The encoder maps an input sequence of symbol 
        representations to a sequence of continuous representations. Given this sequence, the decoder 
        then generates an output sequence of symbols one element at a time.
        
        The attention function maps a query and a set of key-value pairs to an output. The output is 
        computed as a weighted sum of the values, where the weight assigned to each value is computed 
        by a compatibility function of the query with the corresponding key. The model uses multi-head 
        attention, which allows the model to jointly attend to information from different representation 
        subspaces at different positions.
        
        Positional encoding is added to the input embeddings to give the model information about the 
        relative or absolute position of the tokens in the sequence. The positional encodings have the 
        same dimension as the embeddings so they can be summed. The authors used sine and cosine 
        functions of different frequencies for positional encoding.
        """ * 20,  # 重复让文本变长
        "question": "What year was the transformer architecture introduced?",
        "answer": "2017"
    },
    {
        "context": """PagedAttention was introduced by the vLLM team to solve memory management 
        challenges in large language model inference. The key insight is that the KV cache of a 
        request can be stored in non-contiguous memory spaces, similar to how operating systems 
        manage virtual memory with paging. In traditional LLM serving systems, the KV cache for 
        each request is stored in contiguous GPU memory, which leads to significant memory 
        fragmentation and limits the number of requests that can be served concurrently.
        
        The PagedAttention algorithm divides the KV cache into fixed-size blocks, each containing 
        the key-value pairs for a fixed number of tokens. These blocks can be stored in 
        non-contiguous physical memory, and a block table maintains the mapping between logical 
        and physical blocks for each request. This approach virtually eliminates memory 
        fragmentation and enables more efficient memory sharing between requests.
        
        The implementation uses a block size of 16 tokens per block by default. When a request 
        needs more KV cache space, the system allocates new blocks rather than requiring 
        contiguous memory. This allows the system to pack more requests into the same amount 
        of GPU memory, increasing throughput significantly.
        """ * 20,
        "question": "What is the default block size used in PagedAttention?",
        "answer": "16 tokens"
    }
]

def get_gpu_memory():
    return torch.cuda.memory_allocated() / 1024**3  # GB

def run_inference(model, tokenizer, context, question, use_snapkv=False, snapkv_cluster=None):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
    
    input_len = inputs.input_ids.shape[-1]
    print(f"  Input length: {input_len} tokens")
    
    mem_before = get_gpu_memory()
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=1.0,
        )
    
    elapsed = time.time() - start_time
    mem_after = get_gpu_memory()
    
    generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    
    return {
        "answer": generated.strip(),
        "time": elapsed,
        "mem_used_gb": mem_after - mem_before,
        "input_len": input_len
    }

def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cuda",
        use_cache=True,
    )
    model.eval()
    print("Model loaded.\n")

    results = []

    for i, test in enumerate(TEST_CASES):
        print(f"=== Test Case {i+1} ===")
        print(f"Question: {test['question']}")
        print(f"Expected: {test['answer']}")
        
        # Baseline: 完整 KV Cache
        print("\n[Baseline - Full KV Cache]")
        baseline = run_inference(model, tokenizer, test["context"], test["question"])
        print(f"  Answer: {baseline['answer']}")
        print(f"  Time: {baseline['time']:.2f}s")
        print(f"  Memory delta: {baseline['mem_used_gb']:.3f} GB")
        
        torch.cuda.empty_cache()
        
        results.append({
            "test_case": i+1,
            "question": test["question"],
            "expected": test["answer"],
            "baseline": baseline,
        })
        print()

    # 保存结果
    with open("results_baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to results_baseline.json")

if __name__ == "__main__":
    main()
