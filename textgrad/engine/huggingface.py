import os
import platformdirs
from typing import List, Union, Optional
from transformers import pipeline
from transformers.pipelines import TextGenerationPipeline
import base64
import json
from .base import EngineLM, CachedEngine



class ChatHuggingFace(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str,
        system_prompt: str=DEFAULT_SYSTEM_PROMPT,
        use_cache: bool=False,
        **kwargs):

        if use_cache:
            root = platformdirs.user_cache_dir("textgrad")
            cache_path = os.path.join(root, f"cache_hf_{model_string.replace('/', '_')}.db")
            super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt
        self.model = pipeline(model=model_string, **kwargs)

        if isinstance(self.model, TextGenerationPipeline):
            self.is_multimodal = False
        else:
            self.is_multimodal = True
        
        self.model_string = model_string
        self.use_cache = use_cache


    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt: Optional[str]=None, **kwargs):
        if system_prompt is None:
            system_prompt = self.system_prompt

        if self.is_multimodal:
            if not isinstance(content, list):
                content = [content]
            return self._multimodal_generate(content, system_prompt, **kwargs)
        else:
            if isinstance(content, list):
                raise ValueError("This model does not support multimodal input.")
            return self._text_generate(content, system_prompt, **kwargs)


    def _text_generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str]=None, 
        temperature: float=0.1, 
        max_tokens: int=2000, 
        top_p: float=0.99, 
        **kwargs
    ):
        if system_prompt is None:
            system_prompt = self.system_prompt

        if self.use_cache:
            cache_key = system_prompt + prompt
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]}
        ]

        output = self.model(
            messages, 
            temperature=temperature, 
            max_new_tokens=max_tokens, 
            top_p=top_p, 
            **kwargs
        )
        response = output[0]['generated_text'][-1]['content']

        if self.use_cache:
            self._save_cache(cache_key, response)

        return response

    
    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                base64_image = base64.b64encode(item).decode('utf-8')
                formatted_content.append({
                    "type": "image",
                    "base64": base64_image,
                })
            elif isinstance(item, str):
                formatted_content.append({
                    "type": "text",
                    "text": item
                })
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content


    def _multimodal_generate(
        self, 
        content: List[Union[str, bytes]], 
        system_prompt: Optional[str]=None, 
        temperature: float=0.1, 
        max_tokens: int=2000, 
        top_p: float=0.99, 
        **kwargs
    ):
        if system_prompt is None:
            system_prompt = self.system_prompt

        formatted_content = self._format_content(content)

        if self.use_cache:
            cache_key = system_prompt + json.dumps(formatted_content)
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none         
     
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_content}
        ]

        generate_kwargs = {
            "do_sample": True,
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "top_p": top_p,
            **kwargs
        }

        output = self.model(
            text=messages, 
            return_full_text=False, 
            generate_kwargs=generate_kwargs,
        )
        response = output[0]['generated_text']

        if self.use_cache:
            self._save_cache(cache_key, response)
        return response


    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
