# Stable Diffusion Inpainting Finetune
    - Images must be .png
    - To use custom mask images for training, give it the same filename as the original image but with '_mask' appended. Example: (original: "dog_walking_on_two_legs.png" | depth: "dog_walking_on_two_legs_mask.png")
        - Any image without a corresponding mask will have one generated randomly
    - To use filenames as prompts just leave out the 'instance_prompt' argument

# Arguments
    -pretrained_model_name_or_path | Path to pretrained model or model identifier from huggingface.co/models.

    -tokenizer_name | Pretrained tokenizer name or path if not the same as model_name.

    -instance_data_dir | A folder containing the training data of instance images.

    -class_data_dir | A folder containing the training data of class images.

    -instance_prompt | The prompt with identifier specifying the instance.

    -instance_prompt_shuffle_prob | The probability of randomizing the instance prompt.

    -mask_scale_min | The probability of randomizing the instance prompt.

    -mask_scale_max | The probability of randomizing the instance prompt.

    -mask_dropout_prob | The probability of using a blank mask.

    -class_mask_scale_min | The probability of randomizing the instance prompt for the class images.

    -class_mask_scale_max | The probability of randomizing the instance prompt for the class images.

    -class_mask_dropout_prob | The probability of using a blank mask for the class images.

    -instance_prompt_sep_token | The seperator to use to randomize the substrings in the instance prompt.

    -class_prompt | The prompt to specify images in the same class as provided instance images.

    -with_prior_preservation | Flag to add prior preservation loss.

    -use_custom_instance_mask | Flag to add custom mask images.

    -discretize_mask | Whether or not discretize mask (make all values 0 or 1).

    -prior_loss_weight | The weight of prior preservation loss.

    -num_class_images" | Minimal class images for prior preservation loss. If not have enough images, additional images will be sampled with class_prompt.

    -output_dir | The output directory where the model predictions and checkpoints will be written.

    -seed | A seed for reproducible training.

    -resolution" | The resolution for input images, all the images in the train/validation dataset will be resized to this resolution.

    -center_crop | Whether to center crop images before resizing to resolution.

    -train_text_encoder | Whether to train the text encoder.

    -train_batch_size | Batch size (per device) for the training dataloader.

    -sample_batch_size | Batch size (per device) for sampling images.

    -num_train_epochs

    -max_train_steps | Total number of training steps to perform.  If provided, overrides num_train_epochs.

    -gradient_accumulation_steps | "Number of updates steps to accumulate before performing a backward/update pass.

    -gradient_checkpointing | Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.

    -learning_rate | Initial learning rate (after the potential warmup period) to use.

    -scale_lr | Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.

    -lr_scheduler | The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"].

    -lr_warmup_steps | Number of steps for the warmup in the lr scheduler.

    -use_8bit_adam | Whether or not to use 8-bit Adam from bitsandbytes.

    -adam_beta1 | The beta1 parameter for the Adam optimizer.

    -adam_beta2" | The beta2 parameter for the Adam optimizer.

    -adam_weight_decay | Weight decay to use.

    -adam_epsilon | Epsilon value for the Adam optimizer.

    -max_grad_norm | Max gradient norm.

    -push_to_hub | Whether or not to push the model to the Hub.

    -hub_token | The token to use to push to the Model Hub.

    -hub_model_id | The name of the repository to keep in sync with the local `output_dir`.

    -logging_dir | [TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.

    -mixed_precision | Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU.

    -local_rank | For distributed training: local_rank.

    -checkpointing_steps | Save a checkpoint of the training state every X updates. These checkpoints can be used both as final checkpoints in case they are better than the last checkpoint and are suitable for resuming training using `--resume_from_checkpoint`.

    -resume_from_checkpoint | Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.

    -local_rank", type=int, default=-1, help="For distributed training: local_rank.

    -num_workers", type=int, default=1, help="Number of processes to use for data loading.

    -pin_memory", action="store_true", help="Whether or not to pin memory for data loading.

    -persistant_workers", action="store_true", help="Whether or not to use persistent workers.

    -prefetch_factor", type=int, default=2, help="Number of batches to prefetch.

    -drop_incomplete_batches", action="store_true", help="Whether or not to drop incomplete batches. (May help stabilize gradient).