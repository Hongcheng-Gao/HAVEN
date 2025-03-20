import os
from PIL import Image
import numpy as np
import torch
from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from video_chatgpt.model.utils import KeywordsStoppingCriteria
from base import ViLLMBaseModel
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"

class VideoChatGPT(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args['model_path'], model_args['device'])
        assert(
            "model_path" in model_args
            and "device" in model_args
        )
        self.device = model_args["device"]
        self.conv_mode = "video-chatgpt_v1"
        self.vision_tower_name = "openai/clip-vit-large-patch14"

        print("start to load model")
        self.model, self.vision_tower, self.tokenizer, self.image_processor, self.video_token_len = initialize_model(
            os.path.join(model_args["model_path"], "LLaVA-7B-Lightening-v1-1"),
            os.path.join(model_args["model_path"], "Video-ChatGPT-7B/video_chatgpt-7B.bin")
        )
        print('load completed')

        
    def generate(self, instruction, video_path):
        if self.model.get_model().vision_config.use_vid_start_end:
            qs = instruction + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * self.video_token_len + DEFAULT_VID_END_TOKEN
        else:
            qs = instruction + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * self.video_token_len
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = self.tokenizer([prompt])
        video_frames = load_video(video_path)
        image_tensor = self.image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.half().cuda(self.device)
        with torch.no_grad():
            image_forward_outs = self.vision_tower(image_tensor, output_hidden_states=True)
            frame_features = image_forward_outs.hidden_states[-2][:, 1:] # Use second to last layer as in LLaVA
        video_spatio_temporal_features = get_spatio_temporal_features_torch(frame_features)
        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria])
        n_diff_input_output = (input_ids != output_ids[:, :input_ids.shape[1]]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        outputs = outputs.strip().rstrip(stop_str).strip()

        return outputs


def get_spatio_temporal_features_torch(features):
    # Extract the dimensions of the features
    t, s, c = features.shape
    # Compute temporal tokens as the mean along the time axis
    temporal_tokens = torch.mean(features, dim=1)
    padding_size = 100 - t
    if padding_size > 0:
        padding = torch.zeros(padding_size, c, device=features.device)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)
    spatial_tokens = torch.mean(features, dim=0)
    concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0).half()

    return concat_tokens

