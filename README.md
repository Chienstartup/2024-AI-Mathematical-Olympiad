# AI Mathematical Olympiad

- Precision Setting:

  load_in_4bit = True: Uses 4-bit precision, significantly reducing memory usage.</br>
  Other options: 8-bit precision (load_in_8bit = True) or no quantization (default).</br>
  4-bit vs 8-bit: 4-bit offers greater memory savings but may slightly reduce accuracy; 8-bit balances memory savings and precision.


- Quantization Type:

  bnb_4bit_quant_type="nf4": Uses Normal Float 4, optimized for normally distributed weights..</br>
  Other options: Such as "fp4" (Float Point 4)..</br>
  NF4 vs FP4: NF4 is suitable for normally distributed weights, FP4 provides a wider dynamic range.


- Computation Data Type:

  bnb_4bit_compute_dtype=torch.bfloat16: Uses bfloat16 for computations.
  bfloat16 vs float16:

  bfloat16 has a larger exponent range, suitable for large values and gradients in deep learning..</br>
  float16 provides higher precision for small values..</br>
  bfloat16 typically performs better in terms of training stability and hardware compatibility.

- Double Quantization:

  bnb_4bit_use_double_quant=True: Enables double quantization, further compressing the model..</br>
  Without double quantization, one can use load_in_4bit or load_in_8bit alone..</br>
  Double quantization provides additional memory savings but may slightly increase computational overhead.


<h1> Method 1: IA3, Infused Adapter by inhibiting and Amplifying Inner Activations</h1>

  - Characteristics of the IA3 method:

  It fine-tunes the model by injecting adapters into the model's internal activations.
  "Inhibiting and Amplifying" refers to its ability to suppress or enhance certain internal features of the model.
  This method can effectively adapt to new tasks while adjusting very few parameters.

- Advantages of using IA3:

  High parameter efficiency: Only a small number of additional parameters need to be trained.
  Strong adaptability: Can effectively adjust the model to suit specific tasks.
  Computational efficiency: Both training and inference are faster compared to full-parameter fine-tuning.

- This configuration is particularly suitable for:

  Situations requiring quick adaptation to new tasks.</br>
  Environments with limited computational resources.</br>
  Scenarios where maintaining most of the original model's knowledge is necessary.


  <h1> Method 2: QLoRA, Low-Rank Adaptation</h1>

  QLoRA technology: this method combines the advantages of quantization (reduced memory usage) with the efficiency of LoRA (parameter-efficient fine-tuning), enabling effective fine-tuning of large language models with limited resources.

## Explanation of LoRA vs IA3

  Following code uses the LoRA (Low-Rank Adaptation) technique, which differs from the previously discussed IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations).

- LoRA vs IA3:

  LoRA: Adapts pre-trained models by adding low-rank matrices.

  IA3: Adjusts models by inhibiting and amplifying internal activations.

- Main Advantages of LoRA:

  High parameter efficiency: Adds only a small number of trainable parameters.

  Computational efficiency: Fast training and inference.

  Flexibility: Easy to switch or combine different LoRA adapters.

  In summary, this code sets up a PEFT configuration using LoRA technology for a causal language modeling task, primarily adjusting the projection matrices in the attention mechanism. Compared to IA3, LoRA offers a different approach to parameter-efficient fine-tuning, potentially performing better or being easier to adjust for certain tasks.


  <h1> Method 3: p-Tuning</h1>

  P-tuning is a parameter-efficient fine-tuning technique specifically designed for large language models. Here are the main features and working principles of P-tuning:

- Basic Concept:
  P-tuning focuses on optimizing the model's prompts.
  It replaces manually designed discrete prompts with learned continuous virtual tokens.

- Working Mechanism:
  Adds trainable embedding vectors at the beginning of the input sequence.
  These embedding vectors are called "virtual tokens" and their values are optimized during training.
  Virtual tokens act as a soft prompt, guiding the model to generate task-specific outputs.

- Advantages:
  High parameter efficiency: Only a small number of parameters need to be trained.
  Flexibility: Can adapt to different tasks and domains.
  Performance: Can achieve performance comparable to full model fine-tuning on certain tasks.

- Applications:
  Particularly suitable for few-shot and zero-shot learning scenarios.
  Performs well in various natural language processing tasks such as text classification, named entity recognition, etc.

- Comparison with Other Methods:
  Unlike traditional fine-tuning, P-tuning only modifies the input layer without changing other parts of the model.
  Compared to LoRA, P-tuning focuses on optimizing the input layer rather than the internal weights of the model.

- Implementation:
  Usually requires modification of the model's input processing part.
  In Hugging Face's Transformers library, it can be implemented by customizing the tokenizer and model architecture.

  P-tuning represents a new approach of adapting to new tasks by optimizing the input rather than directly modifying model parameters. This method achieves efficient task adaptation while keeping most of the model parameters unchanged.
