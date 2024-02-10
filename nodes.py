import os
import math
import random
import logging
import inspect
import datetime

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf

import torch
import torchvision
import torch.nn.functional as F

from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler

from transformers import CLIPTextModel, CLIPTokenizer

from .animatediff.models.unet import UNet3DConditionModel
from .animatediff.pipelines.pipeline_animation import AnimationPipeline
from .animatediff.utils.util import save_videos_grid, load_diffusers_lora, load_weights
from .animatediff.utils.lora_handler import LoraHandler
from .animatediff.utils.lora import extract_lora_child_module

from .motion_lora import MotionLoraInfo, MotionLoraList

from lion_pytorch import Lion
import comfy.model_management
import comfy.utils
import folder_paths


script_directory = os.path.dirname(os.path.abspath(__file__))
folder_paths.add_model_folder_path("animatediff_models", str(Path(__file__).parent.parent / "models"))
folder_paths.add_model_folder_path("animatediff_models", str(Path(folder_paths.models_dir) / "animatediff_models"))

def create_save_paths(output_dir: str):
    #lora_path = f"{output_dir}/lora"

    directories = [
        output_dir,
        f"{output_dir}/samples",
        f"{output_dir}/sanity_check",
        #lora_path
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    #return lora_path

def do_sanity_check(
    pixel_values: torch.Tensor,  
    output_dir: str = "",
    text_prompt: str = ""
):
    pixel_values, texts = pixel_values.cpu(), text_prompt  
    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
        pixel_value = pixel_value[None, ...]
        text = text
        save_name = f"{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'-{idx}'}.mp4"
        save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{save_name}", rescale=False)
    return(pixel_values)

def sample_noise(latents, noise_strength, use_offset_noise=False):
    b, c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents

def param_optim(model, condition, extra_params=None, is_lora=False, negation=None):
    extra_params = extra_params if len(extra_params.keys()) > 0 else None
    return {
        "model": model,
        "condition": condition,
        'extra_params': extra_params,
        'is_lora': is_lora,
        "negation": negation
    }

def create_optim_params(name='param', params=None, lr=5e-6, extra_params=None):
    params = {
        "name": name,
        "params": params,
        "lr": lr
    }
    if extra_params is not None:
        for k, v in extra_params.items():
            params[k] = v

    return params

def create_optimizer_params(model_list, lr):
    import itertools
    optimizer_params = []

    for optim in model_list:
        model, condition, extra_params, is_lora, negation = optim.values()
        # Check if we are doing LoRA training.
        if is_lora and condition and isinstance(model, list):
            params = create_optim_params(
                params=itertools.chain(*model),
                extra_params=extra_params
            )
            optimizer_params.append(params)
            continue

        if is_lora and condition and not isinstance(model, list):
            for n, p in model.named_parameters():
                if 'lora' in n:
                    params = create_optim_params(n, p, lr, extra_params)
                    optimizer_params.append(params)
            continue

        # If this is true, we can train it.
        if condition:
            for n, p in model.named_parameters():
                should_negate = 'lora' in n and not is_lora
                if should_negate: continue
            
                params = create_optim_params(n, p, lr, extra_params)
                optimizer_params.append(params)

    return optimizer_params

def scale_loras(lora_list: list, scale: float, step=None):
    
    # Assumed enumerator
    if step is not None:
        process_list = range(0, len(lora_list), 1)
    else:
        process_list = lora_list

    for lora_i in process_list:
        if step is not None:
            lora_list[lora_i].scale = scale
        else:
            lora_i.scale = scale

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w") 
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215

    return latents

def get_spatial_latents(
        pixel_values: torch.Tensor, 
        noisy_latents:torch.Tensor, 
        target: torch.Tensor,
    ):
    ran_idx = torch.randint(0, pixel_values.shape[2], (1,)).item()

    noisy_latents_input = None
    target_spatial = None
    
    noisy_latents_input = noisy_latents[:, :, ran_idx, :, :]
    target_spatial = target[:, :, ran_idx, :, :]

    return noisy_latents_input, target_spatial

def create_ad_temporal_loss(
        model_pred: torch.Tensor, 
        loss_temporal: torch.Tensor, 
        target: torch.Tensor
    ):

    beta = 1
    alpha = (beta ** 2 + 1) ** 0.5

    ran_idx = torch.randint(0, model_pred.shape[2], (1,)).item()

    model_pred_decent = alpha * model_pred - beta * model_pred[:, :, ran_idx, :, :].unsqueeze(2)
    target_decent = alpha * target - beta * target[:, :, ran_idx, :, :].unsqueeze(2)

    loss_ad_temporal = F.mse_loss(model_pred_decent.float(), target_decent.float(), reduction="mean")
    loss_temporal = loss_temporal + loss_ad_temporal

    return loss_temporal

class AD_MotionDirector_train:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            #"validation_settings": ("VALIDATION_SETTINGS", ),
            "pipeline": ("PIPELINE", ),
            "lora_name": ("STRING", {"multiline": False, "default": "motiondirectorlora",}),
            "images": ("IMAGE", ),
            "prompt": ("STRING", {"multiline": True, "default": "",}),
            "max_train_steps": ("INT", {"default": 300, "min": 0, "max": 100000, "step": 1}),
            "learning_rate": ("FLOAT", {"default": 5e-4, "min": 0, "max": 10000, "step": 0.00001}),
            "learning_rate_spatial": ("FLOAT", {"default": 1e-4, "min": 0, "max": 10000, "step": 0.00001}),    
            "lora_rank": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 8}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            
            "optimization_method": (
            [   
                'Lion',
                'AdamW',
            ], {
               "default": 'Lion'
            }),
            
            },
            
            }
    
    RETURN_TYPES = ("IMAGE", "ADMDPIPELINE", "LORAINFO")
    RETURN_NAMES =("sanitycheck", "admd_pipeline", "lora_info",)
    FUNCTION = "process"

    CATEGORY = "AD_MotionDirector"

    def process(self, pipeline, images, prompt,  
                lora_name, learning_rate, learning_rate_spatial, 
                lora_rank, seed, optimization_method, max_train_steps):
        with torch.inference_mode(False):
                      
            validation_pipeline = pipeline["validation_pipeline"]
            train_noise_scheduler = pipeline["train_noise_scheduler"]
            train_noise_scheduler_spatial = pipeline["train_noise_scheduler_spatial"]

            unet = pipeline["unet"]
            text_encoder = pipeline["text_encoder"]
            vae = pipeline["vae"]
            tokenizer = pipeline["tokenizer"]

            video_length = images.shape[0]

            input_height, input_width = images.shape[1], images.shape[2]
            images = images * 2.0 - 1.0 #normalize to the expected range (-1, 1)
            pixel_values = images.clone()
            pixel_values = pixel_values.permute(0, 3, 1, 2).unsqueeze(0)#B,H,W,C to B,F,C,H,W

            torch.manual_seed(seed)

            text_prompt = []
            text_prompt.append(prompt)

            device = comfy.model_management.get_torch_device()
            
            cfg_random_null_text = True
            cfg_random_null_text_ratio = 0
  
            scale_lr = False
            lr_warmup_steps = 0
            lr_scheduler = "constant"

            train_batch_size = 1
            adam_beta1 = 0.9
            adam_beta2 = 0.999
            adam_weight_decay = 1e-2
            gradient_accumulation_steps = 1
            gradient_checkpointing = True
            
            is_debug = False

            single_spatial_lora = True
            lora_unet_dropout = 0.1
            target_spatial_modules = ["Transformer3DModel"]
            target_temporal_modules = ["TemporalTransformerBlock"]
            train_sample_validation = False
            

            # validation_inference_steps = validation_settings["inference_steps"]
            # validation_guidance_scale = validation_settings["guidance_scale"]
            # validation_spatial_scale = validation_settings["spatial_scale"]
            # validation_seed = validation_settings["seed"]
            # validation_steps = validation_settings["steps"]
            # validation_steps_tuple = tuple(int(step) for step in validation_settings["steps_tuple"].split(','))  
            # validation_prompt = validation_settings["validation_prompt"]     
            
            name = lora_name
            date_calendar = datetime.datetime.now().strftime("%Y-%m-%d")
            date_time = datetime.datetime.now().strftime("%H-%M-%S")
            folder_name = "debug" if is_debug else name + date_time
            
       
            output_dir = os.path.join(script_directory, "outputs", date_calendar, folder_name)

            if is_debug and os.path.exists(output_dir):
                os.system(f"rm -rf {output_dir}")

            # Make one log on every process with the configuration for debugging.
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                level=logging.INFO,
            )

            # Handle the output folder creation
            #lora_path = create_save_paths(output_dir)
            spatial_lora_path = os.path.join(folder_paths.models_dir,"loras", "trained_spatial", date_calendar, date_time, lora_name) 
            temporal_lora_path = os.path.join(folder_paths.models_dir,"animatediff_motion_lora", date_calendar, date_time, lora_name)
            temporal_lora_base_path = os.path.join(date_calendar, date_time, lora_name)
            
            lora_info = {
                "lora_name": lora_name,
                "lora_rank": lora_rank,
                "spatial_lora_path": spatial_lora_path,
                "temporal_lora_path": temporal_lora_path,
                "temporal_lora_base_path": temporal_lora_base_path
            }

            if optimization_method == "AdamW":   
                print("Using AdamW optimizer for training") 
                optimizer = torch.optim.AdamW
            else:
                print("Using Lion optimizer for training")
                optimizer = Lion
                learning_rate, learning_rate_spatial = map(lambda lr: lr / 10, (learning_rate, learning_rate_spatial))
                adam_weight_decay *= 10

            if scale_lr:
                learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size)

            # Temporal LoRA
            
            # one temporal lora
            lora_manager_temporal = LoraHandler(use_unet_lora=True, unet_replace_modules=target_temporal_modules)
            
            unet_lora_params_temporal, unet_negation_temporal = lora_manager_temporal.add_lora_to_model(
                True, unet, lora_manager_temporal.unet_replace_modules, 0,
                temporal_lora_path, r=lora_rank)

            optimizer_temporal = optimizer(
                create_optimizer_params([param_optim(unet_lora_params_temporal, True, is_lora=True,
                                                    extra_params={**{"lr": learning_rate}}
                                                    )], learning_rate),
                lr=learning_rate,
                betas=(adam_beta1, adam_beta2),
                weight_decay=adam_weight_decay
            )
        
            lr_scheduler_temporal = get_scheduler(
                lr_scheduler,
                optimizer=optimizer_temporal,
                num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
                num_training_steps=max_train_steps * gradient_accumulation_steps,
            )

            # Spatial LoRAs

            unet_lora_params_spatial_list = []
            optimizer_spatial_list = []
            lr_scheduler_spatial_list = []

            lora_manager_spatial = LoraHandler(use_unet_lora=True, unet_replace_modules=target_spatial_modules)
            unet_lora_params_spatial, unet_negation_spatial = lora_manager_spatial.add_lora_to_model(
                True, unet, lora_manager_spatial.unet_replace_modules, lora_unet_dropout,
                spatial_lora_path, r=lora_rank)

            unet_lora_params_spatial_list.append(unet_lora_params_spatial)

            optimizer_spatial = optimizer(
                create_optimizer_params([param_optim(unet_lora_params_spatial, True, is_lora=True,
                                                    extra_params={**{"lr": learning_rate_spatial}}
                                                    )], learning_rate_spatial),
                lr=learning_rate_spatial,
                betas=(adam_beta1, adam_beta2),
                weight_decay=adam_weight_decay
            )

            optimizer_spatial_list.append(optimizer_spatial)

            # Scheduler
            lr_scheduler_spatial = get_scheduler(
                lr_scheduler,
                optimizer=optimizer_spatial,
                num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
                num_training_steps=max_train_steps * gradient_accumulation_steps,
            )
            lr_scheduler_spatial_list.append(lr_scheduler_spatial)
           
            # Train!

        admd_pipeline = {
            "optimizer_temporal": optimizer_temporal,
            "optimizer_spatial_list": optimizer_spatial_list,
            "lr_scheduler_spatial_list": lr_scheduler_spatial_list,
            "lr_scheduler_temporal": lr_scheduler_temporal,
            "text_prompt": text_prompt,
            "unet": unet,
            "text_encoder": text_encoder,
            "vae": vae,
            "tokenizer": tokenizer,
            "pixel_values": pixel_values,
            "train_noise_scheduler": train_noise_scheduler,
            "train_noise_scheduler_spatial": train_noise_scheduler_spatial,
            "validation_pipeline": validation_pipeline,
            "global_step": 0,
        }
        #Data batch sanity check
        
        sanitycheck = do_sanity_check(
            pixel_values,  
            output_dir=output_dir, 
            text_prompt=text_prompt
        )
 
        sanitycheck = sanitycheck.view(*sanitycheck.shape[1:])
        sanitycheck = sanitycheck.permute(1, 2, 3, 0).cpu()
        sanitycheck = (sanitycheck + 1.0) / 2.0
        return (sanitycheck, admd_pipeline, lora_info,)

import folder_paths
class DiffusersLoaderForTraining:
    #@classmethod
    #def IS_CHANGED(s):
    #    return ""
    @classmethod
    def INPUT_TYPES(cls):
        paths = []
        for search_path in folder_paths.get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                for root, subdir, files in os.walk(search_path, followlinks=True):
                    if "model_index.json" in files:
                        paths.append(os.path.relpath(root, start=search_path))

        return {"required":
                {
                "validation_models": ("VALIDATION_MODELS", ),  
                "download_default": ("BOOLEAN", {"default": False},),
                "scheduler": (
            [   
                'DDIMScheduler',
                'DDPMScheduler',
            ], {
               "default": 'DDIMScheduler'
            }),
            "use_xformers": ("BOOLEAN", {"default": False}),
                },
                "optional": {
                 "model": (paths,),
                }
                
            }
    RETURN_TYPES = ("PIPELINE",)

    FUNCTION = "load_checkpoint"

    CATEGORY = "AD_MotionDirector"

    def load_checkpoint(self, download_default, scheduler, use_xformers, validation_models, model=""):
        with torch.inference_mode(False):
            device = comfy.model_management.get_torch_device()
            target_path = os.path.join(folder_paths.models_dir,'diffusers', "stable-diffusion-v1-5")      
            if download_default and model != os.path.exists(target_path):
                from huggingface_hub import snapshot_download
                download_to = os.path.join(folder_paths.models_dir,'diffusers')
                snapshot_download(repo_id="runwayml/stable-diffusion-v1-5", ignore_patterns=["*.safetensors","*.ckpt", "*.pt", "*.png", "*non_ema*", "*safety_checker*", "*fp16*"], 
                                    local_dir=f"{download_to}/stable-diffusion-v1-5", local_dir_use_symlinks=False)   
                model_path = "stable-diffusion-v1-5"
            else:
                model_path = model
            
            for search_path in folder_paths.get_folder_paths("diffusers"):
                if os.path.exists(search_path):
                    path = os.path.join(search_path, model_path)
                    if os.path.exists(path):
                        model_path = path
                        break
          
            config = OmegaConf.load(os.path.join(script_directory, f"configs/training/motion_director/training.yaml"))
            vae          = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
            tokenizer    = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
            
            unet_additional_kwargs = config.unet_additional_kwargs
            unet = UNet3DConditionModel.from_pretrained_2d(
                model_path, subfolder="unet", 
                unet_additional_kwargs=unet_additional_kwargs
            )

            # Load scheduler, tokenizer and models.
            noise_scheduler_kwargs = config.noise_scheduler_kwargs
            noise_scheduler_kwargs.update({"steps_offset": 1})
            noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
            del noise_scheduler_kwargs["steps_offset"]

            if scheduler == "DDPMScheduler":
                print("using DDPMScheduler for training")
                noise_scheduler_kwargs['beta_schedule'] = 'scaled_linear'
                train_noise_scheduler_spatial = DDPMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
                noise_scheduler_kwargs['beta_schedule'] = 'linear'
                train_noise_scheduler = DDPMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
            else:
                print("using DDIMScheduler for training")
                noise_scheduler_kwargs['beta_schedule'] = 'scaled_linear'
                train_noise_scheduler_spatial = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
                noise_scheduler_kwargs['beta_schedule'] = 'linear'
                train_noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
            
            # Freeze all models for LoRA training
            unet.requires_grad_(False)
            vae.requires_grad_(False)
            text_encoder.requires_grad_(False)

            #xformers
            if use_xformers:
                unet.enable_xformers_memory_efficient_attention()

            # Enable gradient checkpointing
            unet.enable_gradient_checkpointing()

            # Move models to GPU
            vae.to(device)
            text_encoder.to(device)
            unet.to(device=device)
            text_encoder.to(device=device)

            # Validation pipeline
            validation_pipeline = AnimationPipeline(
                unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler,
            ).to(device)

            motion_module_path, domain_adapter_path, unet_checkpoint_path = validation_models 

            validation_pipeline = load_weights(
                validation_pipeline, 
                motion_module_path=motion_module_path,
                adapter_lora_path=domain_adapter_path, 
                dreambooth_model_path=unet_checkpoint_path
            )

            validation_pipeline.enable_vae_slicing()         

            pipeline = {
                'validation_pipeline': validation_pipeline,
                'train_noise_scheduler': train_noise_scheduler,
                'train_noise_scheduler_spatial': train_noise_scheduler_spatial,
                'unet': unet,
                'vae': vae,
                'text_encoder': text_encoder,
                'tokenizer': tokenizer
            }

            return (pipeline,)


class ValidationModelSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "motion_module": (folder_paths.get_filename_list("animatediff_models"),),
                "use_adapter_lora": ("BOOLEAN", {"default": True}),
                "use_dreambooth_model": ("BOOLEAN", {"default": False}),                                        
            },
            "optional": {
                "optional_adapter_lora": (folder_paths.get_filename_list("loras"),), 
                "optional_model": (folder_paths.get_filename_list("checkpoints"),),       
            }
        }
    RETURN_TYPES = ("VALIDATION_MODELS",)
    RETURN_NAMES = ("validation_models",)
    FUNCTION = "select_models"

    CATEGORY = "AD_MotionDirector"

    def select_models(self, motion_module, use_adapter_lora, use_dreambooth_model, optional_adapter_lora="", optional_model=""):
        validation_models = []
        motion_module_path = folder_paths.get_full_path("animatediff_models", motion_module)

        if use_adapter_lora:
            adapter_lora_path = folder_paths.get_full_path("loras", optional_adapter_lora)
        else:
            adapter_lora_path = ""        
        if use_dreambooth_model:
            model_path = folder_paths.get_full_path("checkpoints", optional_model)
        else:
            model_path = "" 

        validation_models.append(motion_module_path)
        validation_models.append(adapter_lora_path)
        validation_models.append(model_path)      
        return (validation_models,)

class ValidationSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        "inference_steps": ("INT", {"default": 50, "min": 0, "max": 256, "step": 1}),
        "guidance_scale": ("FLOAT", {"default": 9, "min": 0, "max": 32, "step": 0.1}),
        "spatial_scale": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),                                        
        "validate_at_steps": ("INT", {"default": 50, "min": 0, "max": 10000, "step": 1}),
        "extra_validation_steps": ("STRING", {"default": "2, 25", },),
        "validation_prompt": ("STRING", {"multiline": True, "default": "",}),
        },
          
        }
    RETURN_TYPES = ("VALIDATION_SETTINGS",)
    RETURN_NAMES = ("validation_settings",)
    FUNCTION = "create_validation_settings"

    CATEGORY = "AD_MotionDirector"

    def create_validation_settings(self, inference_steps, guidance_scale, spatial_scale, seed, validate_at_steps, extra_validation_steps, validation_prompt):
        # Create a dictionary with the local variables
        local_vars = locals()
        
        # Filter the dictionary to include only the variables you want
        validation_settings = {
            "inference_steps": local_vars["inference_steps"],
            "guidance_scale": local_vars["guidance_scale"],
            "spatial_scale": local_vars["spatial_scale"],
            "seed": local_vars["seed"],
            "steps": local_vars["validate_at_steps"],
            "steps_tuple": local_vars["extra_validation_steps"],
            "validation_prompt": local_vars["validation_prompt"]
        }
       
        return validation_settings,

class AD_MotionLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_path": ("STRING", {"multiline": False, "default": "",}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            },
            "optional": {
                "prev_motion_lora": ("MOTION_LORA",),
            }
        }
    
    RETURN_TYPES = ("MOTION_LORA",)
    CATEGORY = "AD_MotionDirector"
    FUNCTION = "load_motion_lora"

    def load_motion_lora(self, lora_path: str, strength: float, prev_motion_lora: MotionLoraList=None):
        
        if prev_motion_lora is None:
            prev_motion_lora = MotionLoraList()
        else:
            prev_motion_lora = prev_motion_lora.clone()
        full_lora_path = os.path.join(folder_paths.models_dir,"animatediff_motion_lora",lora_path)
        # check if motion lora with name exists
        if not Path(full_lora_path).is_file():
            raise FileNotFoundError(f"Motion lora not found at {full_lora_path}")
        # create motion lora info to be loaded in AnimateDiff Loader
        lora_name = os.path.basename(lora_path)
        lora_info = MotionLoraInfo(name=lora_path, strength=strength)
        prev_motion_lora.add_lora(lora_info)

        return (prev_motion_lora,)

class SaveMotionDirectorLora:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "admd_pipeline": ("ADMDPIPELINE", ),
                "lora_info": ("LORAINFO", ),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    CATEGORY = "AD_MotionDirector"
    FUNCTION = "save_motion_lora"

    def save_motion_lora(self, admd_pipeline, lora_info):
        with torch.inference_mode(False):
            validation_pipeline = admd_pipeline['validation_pipeline']
            global_step = admd_pipeline['global_step']
            device = comfy.model_management.get_torch_device()

            import copy
            validation_pipeline.to('cpu')  # We do this to prevent VRAM spiking / increase from the new copy
            
            spatial_lora_path = lora_info['spatial_lora_path']
            temporal_lora_path = lora_info['temporal_lora_path']
            lora_name = lora_info['lora_name']
            lora_rank = lora_info['lora_rank']
            temporal_lora_base_path = lora_info['temporal_lora_base_path']

            lora_manager_spatial = LoraHandler(use_unet_lora=True, unet_replace_modules=["Transformer3DModel"])
            lora_manager_spatial.save_lora_weights(
                model=copy.deepcopy(validation_pipeline), 
                save_path=spatial_lora_path, 
                step=global_step,
                use_safetensors=True,
                lora_rank=lora_rank,
                lora_name=lora_name + "_r"+ str(lora_rank) + "_spatial",
            )
            lora_manager_temporal = LoraHandler(use_unet_lora=True, unet_replace_modules=["TemporalTransformerBlock"])
            if lora_manager_temporal is not None:
                lora_manager_temporal.save_lora_weights(
                    model=copy.deepcopy(validation_pipeline), 
                    save_path=temporal_lora_path, 
                    step=global_step,
                    use_safetensors=True,
                    lora_rank=lora_rank,
                    lora_name=lora_name + "_r"+ str(lora_rank) + "_temporal",
                    use_motion_lora_format=True
                )

            #validation_pipeline.to(device)
            final_temporal_lora_name = os.path.join(temporal_lora_base_path, (str(global_step) + "_" + lora_name + "_r"+ str(lora_rank) + "_temporal_unet.safetensors"))
       
            return (final_temporal_lora_name,)

class TrainMotionDirectorLora:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "admd_pipeline": ("ADMDPIPELINE", ),
                "steps": ("INT", {"default": 100, "min": 0, "max": 10000, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("ADMDPIPELINE",)
    CATEGORY = "AD_MotionDirector"
    FUNCTION = "train"

    def train(self, admd_pipeline, steps):
        with torch.inference_mode(False):
            train_noise_scheduler = admd_pipeline["train_noise_scheduler"]
            train_noise_scheduler_spatial = admd_pipeline["train_noise_scheduler_spatial"]
            unet = admd_pipeline["unet"]
            text_encoder = admd_pipeline["text_encoder"]
            vae = admd_pipeline["vae"]
            tokenizer = admd_pipeline["tokenizer"]
            optimizer_temporal = admd_pipeline["optimizer_temporal"]
            optimizer_spatial_list = admd_pipeline["optimizer_spatial_list"]
            lr_scheduler_spatial_list = admd_pipeline["lr_scheduler_spatial_list"]
            lr_scheduler_temporal = admd_pipeline["lr_scheduler_temporal"]
            text_prompt = admd_pipeline["text_prompt"]
            unet = admd_pipeline["unet"]
            pixel_values = admd_pipeline["pixel_values"]

            use_offset_noise = False

            device = comfy.model_management.get_torch_device()
            unet.to(device)
            vae.to(device)
            text_encoder.to(device)
            

            target_spatial_modules = ["Transformer3DModel"]
            target_temporal_modules = ["TemporalTransformerBlock"]

            batch_size = 1
            first_epoch = 0
            gradient_accumulation_steps = 1
            global_step = admd_pipeline["global_step"]
            print(f"global_step: {global_step}")
            max_train_steps = steps

            # Support mixed-precision training
            scaler = torch.cuda.amp.GradScaler()
            
            num_update_steps_per_epoch = math.ceil(batch_size) / gradient_accumulation_steps
            num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

            # Only show the progress bar once on each machine.
            progress_bar = tqdm(range(global_step, max_train_steps))
            progress_bar.set_description("Steps")
            pbar = comfy.utils.ProgressBar(batch_size * num_train_epochs)
            
            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    text_prompt, 
                    max_length=tokenizer.model_max_length, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt"
                ).input_ids.to(device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]

            ### <<<< Training <<<< ###
            for epoch in range(first_epoch, num_train_epochs):
                unet.train()
                
                for step in range(batch_size):
                    spatial_scheduler_lr = 0.0
                    temporal_scheduler_lr = 0.0

                    # Handle Lora Optimizers & Conditions
                    for optimizer_spatial in optimizer_spatial_list:
                        optimizer_spatial.zero_grad(set_to_none=True)

                    if optimizer_temporal is not None:
                        optimizer_temporal.zero_grad(set_to_none=True)
        
                    mask_spatial_lora = random.uniform(0, 1) < 0.2

                    #if cfg_random_null_text:
                    #    text_prompt = [name if random.random() > cfg_random_null_text_ratio else "" for name in text_prompt]

                    # Convert videos to latent space
                    pixel_values = pixel_values.to(device)      

                    # Sample a random timestep for each video
                    timesteps = torch.randint(0, 1000, (1,), device=pixel_values.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    latents = tensor_to_vae_latent(pixel_values, vae)
                    noise = sample_noise(latents, 0, use_offset_noise=use_offset_noise)
                    target = noise

                    with torch.cuda.amp.autocast():
                        if mask_spatial_lora:
                            loras = extract_lora_child_module(unet, target_replace_module=target_spatial_modules)
                            scale_loras(loras, 0.)
                            loss_spatial = None
                        else:
                            loras = extract_lora_child_module(unet, target_replace_module=target_spatial_modules)
                            scale_loras(loras, 1.0)

                            loras = extract_lora_child_module(unet, target_replace_module=target_temporal_modules)
                            if len(loras) > 0:
                                scale_loras(loras, 0.)
                            
                            ### >>>> Spatial LoRA Prediction >>>> ###
                            noisy_latents = train_noise_scheduler_spatial.add_noise(latents, noise, timesteps)
                            noisy_latents_input, target_spatial = get_spatial_latents(
                                pixel_values,  
                                noisy_latents,
                                target,
                            )
                            model_pred_spatial = unet(noisy_latents_input.unsqueeze(2), timesteps,
                                                    encoder_hidden_states=encoder_hidden_states).sample
                            loss_spatial = F.mse_loss(model_pred_spatial[:, :, 0, :, :].float(),
                                                    target_spatial.float(), reduction="mean")
                            
                        loras = extract_lora_child_module(unet, target_replace_module=target_temporal_modules)
                        scale_loras(loras, 1.0)
                        
                        ### >>>> Temporal LoRA Prediction >>>> ###
                        noisy_latents = train_noise_scheduler.add_noise(latents, noise, timesteps)
                        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                        
                        loss_temporal = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        loss_temporal = create_ad_temporal_loss(model_pred, loss_temporal, target)
                            
                        # Backpropagate
                        if not mask_spatial_lora:
                            scaler.scale(loss_spatial).backward(retain_graph=True)
                            scaler.step(optimizer_spatial_list[0])
                                            
                        scaler.scale(loss_temporal).backward()
                        scaler.step(optimizer_temporal)
            
                        lr_scheduler_spatial_list[step].step()
                        spatial_scheduler_lr = lr_scheduler_spatial_list[step].get_lr()[0]
                            
                        if lr_scheduler_temporal is not None:
                            lr_scheduler_temporal.step()
                            temporal_scheduler_lr = lr_scheduler_temporal.get_lr()[0]
                
                    scaler.update()  
                    progress_bar.update(1)
                    pbar.update(1)
                    global_step += 1
                                                    
                    logs = {
                        "Temporal Loss": loss_temporal.detach().item(),
                        "Temporal LR": temporal_scheduler_lr, 
                        "Spatial Loss": loss_spatial.detach().item() if loss_spatial is not None else 0,
                        "Spatial LR": spatial_scheduler_lr
                    }
                    progress_bar.set_postfix(**logs)
                    
                    unet.enable_gradient_checkpointing()

                    if global_step >= max_train_steps:
                        break
            admd_pipeline.update({
                "global_step": global_step,
                "unet": unet,
            })
                
            return (admd_pipeline,)


NODE_CLASS_MAPPINGS = {
    "AD_MotionDirector_train": AD_MotionDirector_train,
    "DiffusersLoaderForTraining": DiffusersLoaderForTraining,
    "ValidationModelSelect": ValidationModelSelect,
    "ValidationSettings": ValidationSettings,
    "AD_MotionLoraLoader": AD_MotionLoraLoader,
    "SaveMotionDirectorLora": SaveMotionDirectorLora,
    "TrainMotionDirectorLora": TrainMotionDirectorLora
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AD_MotionDirector_train": "AD_MotionDirector_train",
    "DiffusersLoaderForTraining": "DiffusersLoaderForTraining",
    "ValidationModelSelect": "ValidationModelSelect",
    "ValidationSettings": "ValidationSettings",
    "AD_MotionLoraLoader": "AD_MotionLoraLoader",
    "SaveMotionDirectorLora": "SaveMotionDirectorLora",
    "TrainMotionDirectorLora": "TrainMotionDirectorLora"
}