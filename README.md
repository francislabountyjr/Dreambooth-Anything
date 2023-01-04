# Dreambooth-Anything
A repository to consolidate stable diffusion finetuning scripts. Train inpainting, depth, v1+, v2+, image variations, image colorization, whatever. Train with optimizations like 8-bit adam and xformers for faster and more memory efficient training. 

# Features
 - Train depth
 - Train inpaint
 - Train on custom image input (image latent concat to noise latent) *idea from [Justin Pinkey](https://www.youtube.com/watch?v=mpMGwQa7J1w)
 - Train on custom conditionings (image embeddings instead of text for example) *idea from [Justin Pinkey](https://www.youtube.com/watch?v=mpMGwQa7J1w)
 - Use filenames as prompts
 - Use bnb 8-bit adam for more memory efficient training
 - Use xformers for more memory efficient training

# Contributing
Pull requests, discussions, requests, suggestions, and critiques are all welcome!

# Disclaimer
This is a combination of a bunch of repos as well as my own code and edits on scripts. I will do my best to give credit where credit is due in the form of comments, licenses, a shout-out on the readme, etc. If I happen to miss giving anyone credit/include a license please email me at labounty3d@gmail.com and I will fix it!

# Shout-Outs
 - Huge thanks to [Hugging Face](https://huggingface.co/) for the diffusers library that makes most of this code possible
 - Huge thanks to [Stable Diffusion](https://stability.ai/) for creating the actual diffusion model and open sourcing it
 - Thanks to [epitaque](https://github.com/epitaque/dreambooth_depth2img) for depth training
 - Another thanks to [Hugging Face](https://github.com/huggingface/diffusers/tree/main/examples/research_projects/dreambooth_inpaint) for inpainting training
 - Shoutout to [EveryDream](https://github.com/victorchall/EveryDream2trainer) for windows venv setup and bnb patch
 - Shoutout to [Justin Pinkey/Lambda Labs](https://github.com/LambdaLabsML/lambda-diffusers) for research in to training with different inputs

# Contact
Reach out to labounty3d@gmail.com with any requests/questions/comments/suggestions/concerns
