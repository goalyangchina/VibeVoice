"""
VibeVoice Gradio Demo - High-Quality Dialogue Generation Interface with Streaming Support
"""

import argparse
import importlib.util
import os
import time
from typing import Iterator
import threading
import numpy as np
import gradio as gr
import librosa
import soundfile as sf
import torch
import traceback

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer
from transformers.utils import logging
from transformers import set_seed

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class VibeVoiceDemo:
    def __init__(self, model_path: str, device: str = "cuda", inference_steps: int = 5):
        """Initialize the VibeVoice demo with model loading."""
        self.model_path = model_path
        self.device = device
        self.inference_steps = inference_steps
        self.is_generating = False  # Track generation state
        self.stop_generation = False  # Flag to stop generation
        self.current_streamer = None  # Track current audio streamer
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("检测到设备设置为CUDA，但当前环境不可用，已自动切换为CPU。")
            self.device = "cpu"
        elif self.device == "auto" and not torch.cuda.is_available():
            logger.warning("检测到 device=auto 但当前环境无可用GPU，已自动切换为CPU。")
            self.device = "cpu"
        self.torch_dtype = self._select_torch_dtype()
        self.attn_implementation = self._select_attention_backend()
        self.load_model()
        self.setup_voice_presets()
        self.load_example_scripts()  # Load example scripts

    def _select_torch_dtype(self):
        """Select a dtype that works across GPU/CPU environments."""
        try:
            if self._using_cuda():
                if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
        except Exception as exc:
            logger.warning(f"自动选择dtype时出现警告，回退到float32: {exc}")
        return torch.float32

    def _select_attention_backend(self):
        """Gracefully pick an attention backend based on availability."""
        if self._using_cuda():
            # Prefer flash attention when available for speed, otherwise use SDPA.
            if importlib.util.find_spec("flash_attn") or importlib.util.find_spec("flash_attn_2"):
                return "flash_attention_2"
            return "sdpa"
        return "eager"

    def _using_cuda(self) -> bool:
        """Check if CUDA is available and selected (including auto)."""
        return (self.device == "auto" or self.device.startswith("cuda")) and torch.cuda.is_available()
        
    def load_model(self):
        """Load the VibeVoice model and processor."""
        print(f"加载处理器与模型：{self.model_path}")
        
        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(
            self.model_path,
        )
        
        # Load model
        device_map = (
            "auto"
            if self.device == "auto" and self._using_cuda()
            else self.device
        )

        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            attn_implementation=self.attn_implementation,
        )
        self.model.eval()
        
        # Use SDE solver by default
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config, 
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        
        if hasattr(self.model.model, 'language_model'):
            print(f"注意力实现：{self.model.model.language_model.config._attn_implementation}，精度：{self.torch_dtype}")
    
    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")
        
        # Check if voices directory exists
        if not os.path.exists(voices_dir):
            print(f"警告：未在 {voices_dir} 找到音色目录")
            self.voice_presets = {}
            self.available_voices = {}
            return
        
        # Scan for all WAV files in the voices directory
        self.voice_presets = {}
        
        # Get all .wav files in the voices directory
        wav_files = [f for f in os.listdir(voices_dir) 
                    if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')) and os.path.isfile(os.path.join(voices_dir, f))]
        
        # Create dictionary with filename (without extension) as key
        for wav_file in wav_files:
            # Remove .wav extension to get the name
            name = os.path.splitext(wav_file)[0]
            # Create full path
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path
        
        # Sort the voice presets alphabetically by name for better UI
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        
        # Filter out voices that don't exist (this is now redundant but kept for safety)
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }
        
        if not self.available_voices:
            raise gr.Error("未找到任何音色预设，请在 demo/voices 目录中添加 .wav 或常见音频文件。")
        
        print(f"已在 {voices_dir} 发现 {len(self.available_voices)} 个音频文件")
        print(f"可用音色：{', '.join(self.available_voices.keys())}")
    
    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file."""
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"读取音频 {audio_path} 时出错：{e}")
            return np.array([])
    
    def generate_podcast_streaming(self, 
                                 num_speakers: int,
                                 script: str,
                                 speaker_1: str = None,
                                 speaker_2: str = None,
                                 speaker_3: str = None,
                                 speaker_4: str = None,
                                 cfg_scale: float = 1.3) -> Iterator[tuple]:
        try:
            
            # Reset stop flag and set generating state
            self.stop_generation = False
            self.is_generating = True
            
            # Validate inputs
            if not script.strip():
                self.is_generating = False
                raise gr.Error("错误：请输入对话脚本。")

            # Defend against common mistake
            script = script.replace("’", "'")
            
            if num_speakers < 1 or num_speakers > 4:
                self.is_generating = False
                raise gr.Error("错误：说话人数量需在 1 到 4 之间。")
            
            # Collect selected speakers
            selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]
            
            # Validate speaker selections
            for i, speaker in enumerate(selected_speakers):
                if not speaker or speaker not in self.available_voices:
                    self.is_generating = False
                    raise gr.Error(f"错误：请为说话人 {i+1} 选择有效的声音。")
            
            # Build initial log
            log = f"🎙️ 正在生成包含 {num_speakers} 位说话人的播客\n"
            log += f"📊 参数：CFG={cfg_scale}，推理步数={self.inference_steps}\n"
            log += f"🎭 说话人：{', '.join(selected_speakers)}\n"
            
            # Check for stop signal
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 已按用户请求停止生成", gr.update(visible=False)
                return
            
            # Load voice samples
            voice_samples = []
            for speaker_name in selected_speakers:
                audio_path = self.available_voices[speaker_name]
                audio_data = self.read_audio(audio_path)
                if len(audio_data) == 0:
                    self.is_generating = False
                    raise gr.Error(f"错误：无法读取 {speaker_name} 的音频示例。")
                voice_samples.append(audio_data)
            
            # log += f"✅ Loaded {len(voice_samples)} voice samples\n"
            
            # Check for stop signal
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 已按用户请求停止生成", gr.update(visible=False)
                return
            
            # Parse script to assign speaker ID's
            lines = script.strip().split('\n')
            formatted_script_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if line already has speaker format
                if line.startswith('Speaker ') and ':' in line:
                    formatted_script_lines.append(line)
                else:
                    # Auto-assign to speakers in rotation
                    speaker_id = len(formatted_script_lines) % num_speakers
                    formatted_script_lines.append(f"Speaker {speaker_id}: {line}")
            
            formatted_script = '\n'.join(formatted_script_lines)
            log += f"📝 已整理脚本，共 {len(formatted_script_lines)} 轮对话\n\n"
            log += "🔄 VibeVoice 流式生成中...\n"
            
            # Check for stop signal before processing
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 已按用户请求停止生成", gr.update(visible=False)
                return
            
            start_time = time.time()
            
            inputs = self.processor(
                text=[formatted_script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Create audio streamer
            audio_streamer = AudioStreamer(
                batch_size=1,
                stop_signal=None,
                timeout=None
            )
            
            # Store current streamer for potential stopping
            self.current_streamer = audio_streamer
            
            # Start generation in a separate thread
            generation_thread = threading.Thread(
                target=self._generate_with_streamer,
                args=(inputs, cfg_scale, audio_streamer)
            )
            generation_thread.start()
            
            # Wait for generation to actually start producing audio
            time.sleep(1)  # Reduced from 3 to 1 second

            # Check for stop signal after thread start
            if self.stop_generation:
                audio_streamer.end()
                generation_thread.join(timeout=5.0)  # Wait up to 5 seconds for thread to finish
                self.is_generating = False
                yield None, "🛑 已按用户请求停止生成", gr.update(visible=False)
                return

            # Collect audio chunks as they arrive
            sample_rate = 24000
            all_audio_chunks = []  # For final statistics
            pending_chunks = []  # Buffer for accumulating small chunks
            chunk_count = 0
            last_yield_time = time.time()
            min_yield_interval = 15 # Yield every 15 seconds
            min_chunk_size = sample_rate * 30 # At least 2 seconds of audio
            
            # Get the stream for the first (and only) sample
            audio_stream = audio_streamer.get_stream(0)
            
            has_yielded_audio = False
            has_received_chunks = False  # Track if we received any chunks at all
            
            for audio_chunk in audio_stream:
                # Check for stop signal in the streaming loop
                if self.stop_generation:
                    audio_streamer.end()
                    break
                    
                chunk_count += 1
                has_received_chunks = True  # Mark that we received at least one chunk
                
                # Convert tensor to numpy
                if torch.is_tensor(audio_chunk):
                    # Convert bfloat16 to float32 first, then to numpy
                    if audio_chunk.dtype == torch.bfloat16:
                        audio_chunk = audio_chunk.float()
                    audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                else:
                    audio_np = np.array(audio_chunk, dtype=np.float32)
                
                # Ensure audio is 1D and properly normalized
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.squeeze()
                
                # Convert to 16-bit for Gradio
                audio_16bit = convert_to_16_bit_wav(audio_np)
                
                # Store for final statistics
                all_audio_chunks.append(audio_16bit)
                
                # Add to pending chunks buffer
                pending_chunks.append(audio_16bit)
                
                # Calculate pending audio size
                pending_audio_size = sum(len(chunk) for chunk in pending_chunks)
                current_time = time.time()
                time_since_last_yield = current_time - last_yield_time
                
                # Decide whether to yield
                should_yield = False
                if not has_yielded_audio and pending_audio_size >= min_chunk_size:
                    # First yield: wait for minimum chunk size
                    should_yield = True
                    has_yielded_audio = True
                elif has_yielded_audio and (pending_audio_size >= min_chunk_size or time_since_last_yield >= min_yield_interval):
                    # Subsequent yields: either enough audio or enough time has passed
                    should_yield = True
                
                if should_yield and pending_chunks:
                    # Concatenate and yield only the new audio chunks
                    new_audio = np.concatenate(pending_chunks)
                    new_duration = len(new_audio) / sample_rate
                    total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                    
                    log_update = log + f"🎵 流式生成：已生成 {total_duration:.1f} 秒（第 {chunk_count} 块）\n"
                    
                    # Yield streaming audio chunk and keep complete_audio as None during streaming
                    yield (sample_rate, new_audio), None, log_update, gr.update(visible=True)
                    
                    # Clear pending chunks after yielding
                    pending_chunks = []
                    last_yield_time = current_time
            
            # Yield any remaining chunks
            if pending_chunks:
                final_new_audio = np.concatenate(pending_chunks)
                total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                log_update = log + f"🎵 最后一段流式输出：累计 {total_duration:.1f} 秒\n"
                yield (sample_rate, final_new_audio), None, log_update, gr.update(visible=True)
                has_yielded_audio = True  # Mark that we yielded audio
            
            # Wait for generation to complete (with timeout to prevent hanging)
            generation_thread.join(timeout=5.0)  # Increased timeout to 5 seconds

            # If thread is still alive after timeout, force end
            if generation_thread.is_alive():
                print("警告：生成线程未在超时时间内结束，将强制停止流式输出")
                audio_streamer.end()
                generation_thread.join(timeout=5.0)

            # Clean up
            self.current_streamer = None
            self.is_generating = False
            
            generation_time = time.time() - start_time
            
            # Check if stopped by user
            if self.stop_generation:
                yield None, None, "🛑 已按用户请求停止生成", gr.update(visible=False)
                return
            
            # Debug logging
            # print(f"Debug: has_received_chunks={has_received_chunks}, chunk_count={chunk_count}, all_audio_chunks length={len(all_audio_chunks)}")
            
            # Check if we received any chunks but didn't yield audio
            if has_received_chunks and not has_yielded_audio and all_audio_chunks:
                # We have chunks but didn't meet the yield criteria, yield them now
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate
                
                final_log = log + f"⏱️ 生成完成，用时 {generation_time:.2f} 秒\n"
                final_log += f"🎵 音频时长：{final_duration:.2f} 秒\n"
                final_log += f"📊 音频块总数：{chunk_count}\n"
                final_log += "✨ 生成成功！完整音频已准备就绪。\n"
                final_log += "💡 想要更好的效果？尝试重新生成或调整 CFG 强度。"
                
                # Yield the complete audio
                yield None, (sample_rate, complete_audio), final_log, gr.update(visible=False)
                return
            
            if not has_received_chunks:
                error_log = log + f"\n❌ 错误：模型未返回任何音频块。耗时 {generation_time:.2f} 秒"
                yield None, None, error_log, gr.update(visible=False)
                return
            
            if not has_yielded_audio:
                error_log = log + f"\n❌ 错误：音频已生成但未完成流式输出。音频块数量：{chunk_count}"
                yield None, None, error_log, gr.update(visible=False)
                return

            # Prepare the complete audio
            if all_audio_chunks:
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate
                
                final_log = log + f"⏱️ 生成完成，用时 {generation_time:.2f} 秒\n"
                final_log += f"🎵 音频时长：{final_duration:.2f} 秒\n"
                final_log += f"📊 音频块总数：{chunk_count}\n"
                final_log += "✨ 生成成功！完整音频已出现在下方的“完整音频”区域。\n"
                final_log += "💡 想要更好的效果？尝试重新生成或调整 CFG 强度。"
                
                # Final yield: Clear streaming audio and provide complete audio
                yield None, (sample_rate, complete_audio), final_log, gr.update(visible=False)
            else:
                final_log = log + "❌ 未生成任何音频。"
                yield None, None, final_log, gr.update(visible=False)

        except gr.Error as e:
            # Handle Gradio-specific errors (like input validation)
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ 输入错误：{str(e)}"
            print(error_msg)
            yield None, None, error_msg, gr.update(visible=False)
            
        except Exception as e:
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ 出现未预期的错误：{str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            yield None, None, error_msg, gr.update(visible=False)
    
    def _generate_with_streamer(self, inputs, cfg_scale, audio_streamer):
        """Helper method to run generation with streamer in a separate thread."""
        try:
            # Check for stop signal before starting generation
            if self.stop_generation:
                audio_streamer.end()
                return
                
            # Define a stop check function that can be called from generate
            def check_stop_generation():
                return self.stop_generation
                
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    'do_sample': False,
                },
                audio_streamer=audio_streamer,
                stop_check_fn=check_stop_generation,  # Pass the stop check function
                verbose=False,  # Disable verbose in streaming mode
                refresh_negative=True,
            )
            
        except Exception as e:
            print(f"生成线程发生错误：{e}")
            traceback.print_exc()
            # Make sure to end the stream on error
            audio_streamer.end()
    
    def stop_audio_generation(self):
        """Stop the current audio generation process."""
        self.stop_generation = True
        if self.current_streamer is not None:
            try:
                self.current_streamer.end()
            except Exception as e:
                print(f"停止流式器时出错：{e}")
        print("🛑 已收到停止生成的请求")
    
    def load_example_scripts(self):
        """Load example scripts from the text_examples directory."""
        examples_dir = os.path.join(os.path.dirname(__file__), "text_examples")
        self.example_scripts = []
        
        # Check if text_examples directory exists
        if not os.path.exists(examples_dir):
            print(f"警告：未在 {examples_dir} 找到示例文本目录")
            return
        
        # Get all .txt files in the text_examples directory
        txt_files = sorted([f for f in os.listdir(examples_dir) 
                          if f.lower().endswith('.txt') and os.path.isfile(os.path.join(examples_dir, f))])
        
        for txt_file in txt_files:
            file_path = os.path.join(examples_dir, txt_file)
            
            import re
            # Check if filename contains a time pattern like "45min", "90min", etc.
            time_pattern = re.search(r'(\d+)min', txt_file.lower())
            if time_pattern:
                minutes = int(time_pattern.group(1))
                if minutes > 15:
                    print(f"Skipping {txt_file}: duration {minutes} minutes exceeds 15-minute limit")
                    continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    script_content = f.read().strip()
                
                # Remove empty lines and lines with only whitespace
                script_content = '\n'.join(line for line in script_content.split('\n') if line.strip())
                
                if not script_content:
                    continue
                
                # Parse the script to determine number of speakers
                num_speakers = self._get_num_speakers_from_script(script_content)
                
                # Add to examples list as [num_speakers, script_content]
                self.example_scripts.append([num_speakers, script_content])
                print(f"Loaded example: {txt_file} with {num_speakers} speakers")
                
            except Exception as e:
                print(f"加载示例脚本 {txt_file} 时出错：{e}")
        
        if self.example_scripts:
            print(f"Successfully loaded {len(self.example_scripts)} example scripts")
        else:
            print("No example scripts were loaded")
    
    def _get_num_speakers_from_script(self, script: str) -> int:
        """Determine the number of unique speakers in a script."""
        import re
        speakers = set()
        
        lines = script.strip().split('\n')
        for line in lines:
            # Use regex to find speaker patterns
            match = re.match(r'^Speaker\s+(\d+)\s*:', line.strip(), re.IGNORECASE)
            if match:
                speaker_id = int(match.group(1))
                speakers.add(speaker_id)
        
        # If no speakers found, default to 1
        if not speakers:
            return 1
        
        # Return the maximum speaker ID + 1 (assuming 0-based indexing)
        # or the count of unique speakers if they're 1-based
        max_speaker = max(speakers)
        min_speaker = min(speakers)
        
        if min_speaker == 0:
            return max_speaker + 1
        else:
            # Assume 1-based indexing, return the count
            return len(speakers)
    

def create_demo_interface(demo_instance: VibeVoiceDemo):
    """Create the Gradio interface with streaming support."""
    
    # Custom CSS for high-end aesthetics with lighter theme
    custom_css = """
    /* Modern light theme with gradients */
    .gradio-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Card styling */
    .settings-card, .generation-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Speaker selection styling */
    .speaker-grid {
        display: grid;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .speaker-item {
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
        border: 1px solid rgba(148, 163, 184, 0.4);
        border-radius: 12px;
        padding: 1rem;
        color: #374151;
        font-weight: 500;
    }
    
    /* Streaming indicator */
    .streaming-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.1); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Queue status styling */
    .queue-status {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        text-align: center;
        font-size: 0.9rem;
        color: #0369a1;
    }
    
    .generate-btn {
        background: linear-gradient(135deg, #059669 0%, #0d9488 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 20px rgba(5, 150, 105, 0.4);
        transition: all 0.3s ease;
    }
    
    .generate-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(5, 150, 105, 0.6);
    }
    
    .stop-btn {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);
        transition: all 0.3s ease;
    }
    
    .stop-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(239, 68, 68, 0.6);
    }
    
    /* Audio player styling */
    .audio-output {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }
    
    .complete-audio-section {
        margin-top: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
    }
    
    /* Text areas */
    .script-input, .log-output {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(148, 163, 184, 0.4) !important;
        border-radius: 12px !important;
        color: #1e293b !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .script-input::placeholder {
        color: #64748b !important;
    }
    
    /* Sliders */
    .slider-container {
        background: rgba(248, 250, 252, 0.8);
        border: 1px solid rgba(226, 232, 240, 0.6);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Labels and text */
    .gradio-container label {
        color: #374151 !important;
        font-weight: 600 !important;
    }
    
    .gradio-container .markdown {
        color: #1f2937 !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2rem; }
        .settings-card, .generation-card { padding: 1rem; }
    }
    
    /* Random example button styling - more subtle professional color */
    .random-btn {
        background: linear-gradient(135deg, #64748b 0%, #475569 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 20px rgba(100, 116, 139, 0.3);
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .random-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(100, 116, 139, 0.4);
        background: linear-gradient(135deg, #475569 0%, #334155 100%);
    }
    """
    
    with gr.Blocks(
        title="VibeVoice - AI播客生成器",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
        )
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎙️ Vibe 播客工坊</h1>
            <p>用 VibeVoice 生成多说话人、长时长的 AI 播客</p>
        </div>
        """)
        
        with gr.Row():
            # Left column - Settings
            with gr.Column(scale=1, elem_classes="settings-card"):
                gr.Markdown("### 🎛️ **播客设置**")
                
                # Number of speakers
                num_speakers = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=2,
                    step=1,
                    label="说话人数量",
                    elem_classes="slider-container"
                )
                
                # Speaker selection
                gr.Markdown("### 🎭 **选择说话人**")
                
                available_speaker_names = list(demo_instance.available_voices.keys())
                # default_speakers = available_speaker_names[:4] if len(available_speaker_names) >= 4 else available_speaker_names
                default_speakers = ['en-Alice_woman', 'en-Carter_man', 'en-Frank_man', 'en-Maya_woman']

                speaker_selections = []
                for i in range(4):
                    default_value = default_speakers[i] if i < len(default_speakers) else None
                    speaker = gr.Dropdown(
                        choices=available_speaker_names,
                        value=default_value,
                        label=f"说话人 {i+1}",
                        visible=(i < 2),  # Initially show only first 2 speakers
                        elem_classes="speaker-item"
                    )
                    speaker_selections.append(speaker)
                
                # Advanced settings
                gr.Markdown("### ⚙️ **高级设置**")
                
                # Sampling parameters (contains all generation settings)
                with gr.Accordion("生成参数", open=False):
                    cfg_scale = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.3,
                        step=0.05,
                        label="CFG 强度（文本约束）",
                        # info="Higher values increase adherence to text",
                        elem_classes="slider-container"
                    )
                
            # Right column - Generation
            with gr.Column(scale=2, elem_classes="generation-card"):
                gr.Markdown("### 📝 **输入脚本**")
                
                script_input = gr.Textbox(
                    label="对话脚本",
                    placeholder="""在此输入对话或旁白脚本，格式可参考：

Speaker 0: 欢迎来到今天的播客！
Speaker 1: 感谢邀请，我很期待讨论...

也可以直接粘贴文本，系统会自动轮流分配说话人。""",
                    lines=12,
                    max_lines=20,
                    elem_classes="script-input"
                )
                
                # Button row with Random Example on the left and Generate on the right
                with gr.Row():
                    # Random example button (now on the left)
                    random_example_btn = gr.Button(
                        "🎲 随机示例",
                        size="lg",
                        variant="secondary",
                        elem_classes="random-btn",
                        scale=1  # Smaller width
                    )
                    
                    # Generate button (now on the right)
                    generate_btn = gr.Button(
                        "🚀 生成播客",
                        size="lg",
                        variant="primary",
                        elem_classes="generate-btn",
                        scale=2  # Wider than random button
                    )
                
                # Stop button
                stop_btn = gr.Button(
                    "🛑 停止生成",
                    size="lg",
                    variant="stop",
                    elem_classes="stop-btn",
                    visible=False
                )
                
                # Streaming status indicator
                streaming_status = gr.HTML(
                    value="""
                    <div style="background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
                                border: 1px solid rgba(34, 197, 94, 0.3); 
                                border-radius: 8px; 
                                padding: 0.75rem; 
                                margin: 0.5rem 0;
                                text-align: center;
                                font-size: 0.9rem;
                                color: #166534;">
                        <span class="streaming-indicator"></span>
                        <strong>实时流式</strong> - 正在实时生成并播放音频
                    </div>
                    """,
                    visible=False,
                    elem_id="streaming-status"
                )
                
                # Output section
                gr.Markdown("### 🎵 **生成结果**")
                
                # Streaming audio output (outside of tabs for simpler handling)
                audio_output = gr.Audio(
                    label="实时流式音频",
                    type="numpy",
                    elem_classes="audio-output",
                    streaming=True,  # Enable streaming mode
                    autoplay=True,
                    show_download_button=False,  # Explicitly show download button
                    visible=True
                )
                
                # Complete audio output (non-streaming)
                complete_audio_output = gr.Audio(
                    label="完整播客（生成后可下载）",
                    type="numpy",
                    elem_classes="audio-output complete-audio-section",
                    streaming=False,  # Non-streaming mode
                    autoplay=False,
                    show_download_button=True,  # Explicitly show download button
                    visible=False  # Initially hidden, shown when audio is ready
                )
                
                gr.Markdown("""
                *💡 **流式播放**：生成过程中实时播放，可能偶有停顿  
                *💡 **完整音频**：生成完成后会在下方展示并支持下载*
                """)
                
                # Generation log
                log_output = gr.Textbox(
                    label="生成日志",
                    lines=8,
                    max_lines=15,
                    interactive=False,
                    elem_classes="log-output"
                )
        
        def update_speaker_visibility(num_speakers):
            updates = []
            for i in range(4):
                updates.append(gr.update(visible=(i < num_speakers)))
            return updates
        
        num_speakers.change(
            fn=update_speaker_visibility,
            inputs=[num_speakers],
            outputs=speaker_selections
        )
        
        # Main generation function with streaming
        def generate_podcast_wrapper(num_speakers, script, *speakers_and_params):
            """Wrapper function to handle the streaming generation call."""
            try:
                # Extract speakers and parameters
                speakers = speakers_and_params[:4]  # First 4 are speaker selections
                cfg_scale = speakers_and_params[4]   # CFG scale
                
                # Clear outputs and reset visibility at start
                yield None, gr.update(value=None, visible=False), "🎙️ 开始生成...", gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
                
                # The generator will yield multiple times
                final_log = "开始生成..."
                
                for streaming_audio, complete_audio, log, streaming_visible in demo_instance.generate_podcast_streaming(
                    num_speakers=int(num_speakers),
                    script=script,
                    speaker_1=speakers[0],
                    speaker_2=speakers[1],
                    speaker_3=speakers[2],
                    speaker_4=speakers[3],
                    cfg_scale=cfg_scale
                ):
                    final_log = log
                    
                    # Check if we have complete audio (final yield)
                    if complete_audio is not None:
                        # Final state: clear streaming, show complete audio
                        yield None, gr.update(value=complete_audio, visible=True), log, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                    else:
                        # Streaming state: update streaming audio only
                        if streaming_audio is not None:
                            yield streaming_audio, gr.update(visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True)
                        else:
                            # No new audio, just update status
                            yield None, gr.update(visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True)

            except Exception as e:
                error_msg = f"❌ 封装器发生严重错误：{str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                # Reset button states on error
                yield None, gr.update(value=None, visible=False), error_msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        def stop_generation_handler():
            """Handle stopping generation."""
            demo_instance.stop_audio_generation()
            # Return values for: log_output, streaming_status, generate_btn, stop_btn
            return "🛑 已停止生成。", gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        # Add a clear audio function
        def clear_audio_outputs():
            """Clear both audio outputs before starting new generation."""
            return None, gr.update(value=None, visible=False)

        # Connect generation button with streaming outputs
        generate_btn.click(
            fn=clear_audio_outputs,
            inputs=[],
            outputs=[audio_output, complete_audio_output],
            queue=False
        ).then(
            fn=generate_podcast_wrapper,
            inputs=[num_speakers, script_input] + speaker_selections + [cfg_scale],
            outputs=[audio_output, complete_audio_output, log_output, streaming_status, generate_btn, stop_btn],
            queue=True  # Enable Gradio's built-in queue
        )
        
        # Connect stop button
        stop_btn.click(
            fn=stop_generation_handler,
            inputs=[],
            outputs=[log_output, streaming_status, generate_btn, stop_btn],
            queue=False  # Don't queue stop requests
        ).then(
            # Clear both audio outputs after stopping
            fn=lambda: (None, None),
            inputs=[],
            outputs=[audio_output, complete_audio_output],
            queue=False
        )
        
        # Function to randomly select an example
        def load_random_example():
            """Randomly select and load an example script."""
            import random
            
            # Get available examples
            if hasattr(demo_instance, 'example_scripts') and demo_instance.example_scripts:
                example_scripts = demo_instance.example_scripts
            else:
                # Fallback to default
                example_scripts = [
                    [2, "Speaker 0: 欢迎体验我们的 AI 播客生成！\nSpeaker 1: 很高兴来到这里，一起聊聊精彩话题。"]
                ]
            
            # Randomly select one
            if example_scripts:
                selected = random.choice(example_scripts)
                num_speakers_value = selected[0]
                script_value = selected[1]
                
                # Return the values to update the UI
                return num_speakers_value, script_value
            
            # Default values if no examples
            return 2, ""
        
        # Connect random example button
        random_example_btn.click(
            fn=load_random_example,
            inputs=[],
            outputs=[num_speakers, script_input],
            queue=False  # Don't queue this simple operation
        )
        
        # Add usage tips
        gr.Markdown("""
        ### 💡 **使用小贴士**
        
        - 点击 **🚀 生成播客** 开始生成音频
        - **实时流式** 会在生成过程中持续播放（可能会有短暂停顿）
        - **完整音频** 会在生成结束后显示可下载的成品
        - 生成过程中可随时点击 **🛑 停止生成** 终止任务
        - 顶部绿色指示灯会实时反馈生成状态
        """)
        
        # Add example scripts
        gr.Markdown("### 📚 **示例脚本**")
        
        # Use dynamically loaded examples if available, otherwise provide a default
        if hasattr(demo_instance, 'example_scripts') and demo_instance.example_scripts:
            example_scripts = demo_instance.example_scripts
        else:
            # Fallback to a simple default example if no scripts loaded
            example_scripts = [
                [1, "Speaker 1: 欢迎体验 VibeVoice AI 播客！这是一段展示自然语音合成效果的示例。"]
            ]
        
        gr.Examples(
            examples=example_scripts,
            inputs=[num_speakers, script_input],
            label="点击直接填充示例："
        )

    return interface


def convert_to_16_bit_wav(data):
    # Check if data is a tensor and move to cpu
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    # Ensure data is numpy array
    data = np.array(data)

    # Normalize to range [-1, 1] if it's not already
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    
    # Scale to 16-bit integer range
    data = (data * 32767).astype(np.int16)
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Gradio 演示")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/tmp/vibevoice-model",
        help="VibeVoice 模型目录路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备（如 cuda、cuda:0 或 cpu）",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=10,
        help="DDPM 推理步数（不对外暴露的内部参数）",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="通过 Gradio 创建可共享链接",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Demo 服务端口",
    )
    
    return parser.parse_args()


def main():
    """Main function to run the demo."""
    args = parse_args()
    
    set_seed(42)  # Set a fixed seed for reproducibility

    print("🎙️ 正在初始化带流式播放的 VibeVoice 演示...")
    
    # Initialize demo instance
    demo_instance = VibeVoiceDemo(
        model_path=args.model_path,
        device=args.device,
        inference_steps=args.inference_steps
    )
    
    # Create interface
    interface = create_demo_interface(demo_instance)
    
    print(f"🚀 即将启动 Demo，端口 {args.port}")
    print(f"📁 模型路径：{args.model_path}")
    print(f"🎭 可用音色数量：{len(demo_instance.available_voices)}")
    print("🔴 流式模式：已启用")
    print("🔒 会话隔离：已启用")
    
    # Launch the interface
    try:
        interface.queue(
            max_size=20,  # Maximum queue size
            default_concurrency_limit=1  # Process one request at a time
        ).launch(
            share=args.share,
            # server_port=args.port,
            server_name="0.0.0.0" if args.share else "127.0.0.1",
            show_error=True,
            show_api=False  # Hide API docs for cleaner interface
        )
    except KeyboardInterrupt:
        print("\n🛑 收到中断信号，正在优雅退出...")
    except Exception as e:
        print(f"❌ 服务启动失败：{e}")
        raise


if __name__ == "__main__":
    main()
