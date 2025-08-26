import argparse
import json
import os

from quickstart_advanced import add_llm_args, setup_llm

from tensorrt_llm.inputs import default_multimodal_input_loader
from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY
from tensorrt_llm.tools.importlib_utils import import_custom_module_from_dir

example_medias_and_prompts = {
    "image": {
        "media": [
            "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
            "https://huggingface.co/datasets/Sayali9141/traffic_signal_images/resolve/main/61.jpg",
        ],
        "prompt": [
            "Describe the natural environment in the image.",
            "Describe the object and the weather condition in the image.",
            "Describe the traffic condition on the road in the image.",
        ]
    },
    "multi_turn_image": {
        "system_prompt": "You are a helpful AI assistant that can analyze images and answer questions about them.",
        "conversations": [
            {
                "media": "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png",
                "user_questions": [
                    "What do you see in this image?",
                    "What colors are prominent in this scene?",
                    "How would you describe the mood of this image?"
                ]
            },
            {
                "media": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
                "user_questions": [
                    "What is the main object in this image?",
                    "What is the weather condition shown?",
                    "How does the lighting affect the scene?"
                ]
            }
        ]
    },
    "video": {
        "media": [
            "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/OAI-sora-tokyo-walk.mp4",
            "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/world.mp4",
        ],
        "prompt": [
            "Tell me what you see in the video briefly.",
            "Describe the scene in the video briefly.",
        ]
    },
    "audio": {
        "media": [
            "https://huggingface.co/microsoft/Phi-4-multimodal-instruct/resolve/main/examples/what_is_the_traffic_sign_in_the_image.wav",
            "https://huggingface.co/microsoft/Phi-4-multimodal-instruct/resolve/main/examples/what_is_shown_in_this_image.wav",
        ],
        "prompt": [
            "Transcribe the audio clip into text, please don't add other text.",
            "Transcribe the audio clip into text, please don't add other text.",
        ]
    },
    "image_audio": {
        "media": [
            [
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
                "https://huggingface.co/microsoft/Phi-4-multimodal-instruct/resolve/main/examples/what_is_shown_in_this_image.wav"
            ],
            [
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
                "https://huggingface.co/microsoft/Phi-4-multimodal-instruct/resolve/main/examples/what_is_shown_in_this_image.wav"
            ],
        ],
        "prompt": [
            "Describe the scene in the image briefly.",
            "",
        ]
    },
    "multiple_image": {
        "media": [
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
            "https://huggingface.co/datasets/Sayali9141/traffic_signal_images/resolve/main/61.jpg",
        ],
        "prompt": ["Describe the difference between the two images."],
    },
    "mixture_text_image": {
        "media": [
            [],
            [
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
            ],
        ],
        "prompt": [
            "Who invented the internet?",
            "Describe the scene in the image briefly.",
        ],
    },
}


def add_multimodal_args(parser):
    parser.add_argument(
        "--model_type",
        type=str,
        choices=MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_model_types(),
        help="Model type as specified in the HuggingFace model config.")
    parser.add_argument("--modality",
                        type=str,
                        choices=[
                            "image", "video", "audio", "image_audio",
                            "multiple_image", "mixture_text_image"
                        ],
                        default="image",
                        help="Media type being used for inference.")
    parser.add_argument("--media",
                        type=str,
                        nargs="+",
                        help="A single or a list of media filepaths / urls.")
    parser.add_argument("--num_frames",
                        type=int,
                        default=8,
                        help="The number of video frames to be sampled.")
    parser.add_argument("--image_format",
                        type=str,
                        choices=["pt", "pil"],
                        default="pt",
                        help="The format of the image.")
    parser.add_argument("--device",
                        type=str,
                        default="cpu",
                        help="The device to have the input on.")
    parser.add_argument(
        "--custom_module_dirs",
        type=str,
        nargs="+",
        default=None,
        help=
        ("Paths to an out-of-tree model directory which should be imported."
         " This is useful to load a custom model. The directory should have a structure like:"
         " <model_name>"
         " ├── __init__.py"
         " ├── <model_name>.py"
         " └── <sub_dirs>"))
    parser.add_argument("--sequential_mode",
                        action='store_true',
                        help="Run requests one by one to observe KV cache reuse (only for image modality for now).")
    parser.add_argument("--multi_turn_mode",
                        action='store_true',
                        help="Run multi-turn conversations where each request builds on previous ones (only for image modality).")
    return parser


def add_lora_args(parser):
    parser.add_argument("--load_lora",
                        default=False,
                        action='store_true',
                        help="Whether to load the LoRA model.")
    parser.add_argument("--auto_model_name",
                        type=str,
                        default=None,
                        help="The auto model name in TRTLLM repo.")
    return parser


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multimodal models with the PyTorch workflow.")
    parser = add_llm_args(parser)
    parser = add_multimodal_args(parser)
    parser = add_lora_args(parser)
    args = parser.parse_args()

    # Enable KV cache reuse for sequential mode with image modality
    if (args.sequential_mode or args.multi_turn_mode) and args.modality == "image":
        args.disable_kv_cache_reuse = False
        if args.sequential_mode:
            print("Sequential mode enabled: KV cache reuse will be enabled for image modality")
        if args.multi_turn_mode:
            print("Multi-turn mode enabled: KV cache reuse will be enabled for image modality")
    else:
        args.disable_kv_cache_reuse = True  # kv cache reuse does not work for multimodal, force overwrite
    
    if args.kv_cache_fraction is None:
        args.kv_cache_fraction = 0.6  # lower the default kv cache fraction for multimodal

    return args


def main():
    args = parse_arguments()
    if args.custom_module_dirs is not None:
        for custom_module_dir in args.custom_module_dirs:
            try:
                import_custom_module_from_dir(custom_module_dir)
            except Exception as e:
                print(
                    f"Failed to import custom module from {custom_module_dir}: {e}"
                )
                raise e

    lora_config = None
    if args.load_lora:
        assert args.auto_model_name is not None, "Please provide the auto model name to load LoRA config."
        import importlib
        models_module = importlib.import_module('tensorrt_llm._torch.models')
        model_class = getattr(models_module, args.auto_model_name)
        lora_config = model_class.lora_config(args.model_dir)
        # For stability - explicitly set the LoRA GPU cache & CPU cache to have space for 2 adapters
        lora_config.max_loras = 2
        lora_config.max_cpu_loras = 2

    llm, sampling_params = setup_llm(args, lora_config=lora_config)

    image_format = args.image_format
    if args.model_type is not None:
        model_type = args.model_type
    else:
        model_type = json.load(
            open(os.path.join(str(llm._hf_model_dir),
                              'config.json')))['model_type']
    assert model_type in MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_model_types(), \
        f"Unsupported model_type: {model_type} found!\n" \
        f"Supported types: {MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_model_types()}"

    if args.sequential_mode and args.modality == "image":
        # For sequential mode, use only the first image and prompt, but run it multiple times
        single_prompt = args.prompt[0] if isinstance(args.prompt, list) else args.prompt
        single_media = args.media[0] if isinstance(args.media, list) else args.media
        
        num_repetitions = 5  # Run the same case 5 times to observe KV cache reuse
        print(f"\n=== Sequential Mode: Processing same case {num_repetitions} times ===")
        print(f"Image: {single_media}")
        print(f"Prompt: {single_prompt}")
        print("This mode allows observation of KV cache reuse when the same image/prompt is processed repeatedly.\n")
        
        outputs = []
        for i in range(num_repetitions):
            print(f"Processing request {i+1}/{num_repetitions}...")
            
            # Create input for single request (same image and prompt each time)
            single_input = default_multimodal_input_loader(
                tokenizer=llm.tokenizer,
                model_dir=str(llm._hf_model_dir),
                model_type=model_type,
                modality=args.modality,
                prompts=[single_prompt],
                media=[single_media],
                image_data_format=image_format,
                num_frames=args.num_frames,
                device=args.device
            )
            
            lora_request = None
            if args.load_lora:
                lora_request = model_class.lora_request(1, args.modality, llm._hf_model_dir)
            
            # Generate for single request
            single_output = llm.generate(
                single_input,
                sampling_params,
                lora_request=lora_request,
            )
            
            outputs.extend(single_output)
            
            # Print result immediately
            generated_text = single_output[0].outputs[0].text
            print(f"[{i}] Generated text: {generated_text!r}")
            print("-" * 80)
        
        print(f"\nCompleted {len(outputs)} sequential requests with the same image/prompt.")
    
    elif args.multi_turn_mode and args.modality == "image":
        print(f"\n=== Multi-Turn Mode: Processing conversation with KV cache reuse ===")
        print("Each request builds on the previous conversation history for maximum KV cache reuse.\n")
        
        # Get multi-turn conversation data
        if args.prompt is None and args.media is None:
            # Use default multi-turn example above
            multi_turn_data = example_medias_and_prompts["multi_turn_image"]
            system_prompt = multi_turn_data["system_prompt"]
            conversations = multi_turn_data["conversations"]
        else:
            # For custom multi-turn, we'll need to structure the data differently
            # For now, use a simple approach with the first image and multiple questions
            system_prompt = "You are a helpful AI assistant that can analyze images and answer questions about them."
            single_media = args.media[0] if isinstance(args.media, list) else args.media
            conversations = [{
                "media": single_media,
                "user_questions": args.prompt if isinstance(args.prompt, list) else [args.prompt]
            }]
        
        outputs = []
        
        for conv_idx, conversation in enumerate(conversations):
            print(f"\n--- Conversation {conv_idx + 1} ---")
            print(f"Image: {conversation['media']}")

            conversation_history = []
            
            for q_idx, user_question in enumerate(conversation['user_questions']):
                print(f"\nProcessing turn {q_idx + 1}/{len(conversation['user_questions'])}...")
                
                # Build the full conversation context
                if q_idx == 0:
                    # First question in this conversation
                    full_prompt = f"{system_prompt}\n\nUser: {user_question}\nAssistant:"
                else:
                    # Subsequent questions - include previous Q&A
                    full_prompt = f"{system_prompt}\n\n"
                    for hist_item in conversation_history:
                        full_prompt += f"User: {hist_item['question']}\nAssistant: {hist_item['response']}\n\n"
                    full_prompt += f"User: {user_question}\nAssistant:"
                    print(f'full prompt in conversation, {full_prompt}')
                
                # Create input for this turn
                turn_input = default_multimodal_input_loader(
                    tokenizer=llm.tokenizer,
                    model_dir=str(llm._hf_model_dir),
                    model_type=model_type,
                    modality=args.modality,
                    prompts=[full_prompt],
                    media=[conversation['media']],
                    image_data_format=image_format,
                    num_frames=args.num_frames,
                    device=args.device
                )
                
                lora_request = None
                if args.load_lora:
                    lora_request = model_class.lora_request(1, args.modality, llm._hf_model_dir)
                
                # Generate response for this turn
                turn_output = llm.generate(
                    turn_input,
                    sampling_params,
                    lora_request=lora_request,
                )
                
                outputs.extend(turn_output)
                
                # Get the generated response
                generated_text = turn_output[0].outputs[0].text
                
                # Store in conversation history
                conversation_history.append({
                    'question': user_question,
                    'response': generated_text
                })
                
                # Print result
                print(f"User: {user_question}")
                print(f"Assistant: {generated_text}")
                print("-" * 80)
        
        print(f"\nCompleted multi-turn conversation with {len(outputs)} total turns.")
           
    else:
        # set prompts and media to example prompts and images if they are not provided
        if args.prompt is None:
            args.prompt = example_medias_and_prompts[args.modality]["prompt"]
        if args.media is None:
            args.media = example_medias_and_prompts[args.modality]["media"]
        inputs = default_multimodal_input_loader(tokenizer=llm.tokenizer,
                                                model_dir=str(llm._hf_model_dir),
                                                model_type=model_type,
                                                modality=args.modality,
                                                prompts=args.prompt,
                                                media=args.media,
                                                image_data_format=image_format,
                                                num_frames=args.num_frames,
                                                device=args.device)

        lora_request = None
        if args.load_lora:
            lora_request = model_class.lora_request(len(inputs), args.modality,
                                                    llm._hf_model_dir)

        outputs = llm.generate(
            inputs,
            sampling_params,
            lora_request=lora_request,
        )

        for i, output in enumerate(outputs):
            prompt = args.prompt[i]
            generated_text = output.outputs[0].text
            print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
