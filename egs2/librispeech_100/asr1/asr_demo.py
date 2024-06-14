import torch
import argparse
from espnet2.bin.asr_inference import Speech2Text
from espnet2.torch_utils import model_summary
import humanfriendly
import logging
import pandas as pd
import soundfile
import librosa.display
#from IPython.display import display, Audio
import matplotlib.pyplot as plt
import time
import string
import openai
#import tensorrt as trt
import torch.onnx
from torchsummary import summary
import onnxruntime as rt
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit


class TrtModel:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

    class HostDeviceMem:
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

    def allocate_buffers(self, input_shapes,output_shapes):
      
        inputs = []
        outputs = []
        bindings = []
        stream = self.stream

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            tensor_mode = self.engine.get_tensor_mode(tensor_name)

            if tensor_mode == trt.TensorIOMode.INPUT:
                shape = input_shapes[0]
                engine_dims = self.engine.get_tensor_shape(tensor_name)
               
                if len(engine_dims) == 3 and len(shape) == 2:
                    shape = (1, *shape)
                if len(engine_dims) != len(shape):
                    raise ValueError(f"Shape mismatch for tensor {tensor_name}: Engine expects {len(engine_dims)} dimensions, but got {len(shape)} dimensions.")


                size = trt.volume(shape)
                self.context.set_input_shape(tensor_name, shape)
            else:
                min_shape, opt_shape, max_shape = self.engine.get_tensor_profile_shape(tensor_name, 0)
                out_shape = output_shapes  
                size = trt.volume(out_shape)
     
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
          
            bindings.append(int(device_mem))
         
            if tensor_mode == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream    

    def infer(self, input_tensors,output_shapes):
        inputs, outputs, bindings, stream = self.allocate_buffers([input_tensor.shape for input_tensor in input_tensors],output_shapes)
       
        for i, input_tensor in enumerate(input_tensors):
            input_data = input_tensor.cpu().numpy().ravel()
            np.copyto(inputs[i].host, input_data)
            cuda.memcpy_htod_async(inputs[i].device, inputs[i].host, stream)
        

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(tensor_name, bindings[i])
        
      
        success = self.context.execute_async_v3(stream_handle=stream.handle)
        if success :
            print("TRT INFER DONE")
        stream.synchronize()
        

        for output in outputs:
            cuda.memcpy_dtoh(output.host, output.device)
        for i, output in enumerate(outputs):
            output_tensors = torch.tensor(output.host).reshape(output_shapes)
        
        return output_tensors


def text_normalizer(text):
    text = text.upper()
    return text.translate(str.maketrans('', '', string.punctuation))

def main(args):
    print("======================================")
    print("======================================")
    print("========   Demo Initialize   =========")
    print("======================================")
    print("======================================")
    model_path = args.model_path
    model_config_path = args.model_config_path
    inference_config = args.inference_config
    device = args.device


    ## model hyper param
    ctc_weight = args.ctc_weight
    beam_size = args.beam_size
    batch_size = args.batch_size
    nbest = args.nbest

    ## load model 
    stt = Speech2Text(
        asr_train_config = model_config_path,
        asr_model_file = model_path,
        batch_size = batch_size,
        beam_size = beam_size,
        ctc_weight = ctc_weight,
        device = device
    )
    ############GPT 
    

    ## TRT 
    trt_file_path = 'model_encoder.trt'  # Path to your .trt file
    trt_model = TrtModel(trt_file_path)

    # data loader
    i =  1

    #file_name = "./demo_data/asr_demo_test_"+str(i+1)+".wav"
    file_name = "./demo_data/test_5s.wav"
    speech, rate = soundfile.read(file_name)
    print(rate)
    print(len(speech))
    assert rate == 16000, "mismatch in sampling rate"
    print("infer start")


    # data processing 
    test_speech = torch.tensor(speech).to(stt.device)
    test_speech = test_speech.unsqueeze(0).to(getattr(torch,stt.dtype))
    test_lengths = torch.tensor([test_speech.size(1)], dtype=torch.int)  
    batch = {"speech":test_speech,"speech_lengths":test_lengths}

    test_speech , test_lengths = stt.asr_model.frontend(test_speech,test_lengths)
    test_speech , test_lengths = stt.asr_model.normalize(test_speech,test_lengths)


    print(test_lengths)
    b,n,c = test_speech.size()
    out_n = int(round((n - 5) / 4))

    output_shapes = [1,out_n,160]
    print(output_shapes)
    start_time = time.time()
    enc = trt_model.infer(test_speech,output_shapes)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"TRT ASR Encoder Elapse tiem : {elapsed_time:.4f} second")

    enc = enc.to(stt.device)

    #encoding
    #start_time = time.time()
    #dummy = stt.asr_model.encoder(test_speech, test_lengths)
    #print(dummy[0].size())
    #end_time = time.time()
    #elapsed_time = end_time - start_time
    #print(f"Naive ASR Encoder Elapse tiem : {elapsed_time:.4f} second")


    if isinstance(enc,tuple):
        enc = enc[0]
    intermediate_outs = None
    if isinstance(enc, tuple):
        intermediate_outs = enc[1]
        enc = enc[0]
        assert len(enc) == 1, len(enc)
    results = stt._decode_single_sample(enc[0])

    if intermediate_outs is not None:
        encoder_interctc_res = stt._decode_interctc(intermediate_outs)
        results = (results, encoder_interctc_res)

    # Result 
    text, *_ = results[0]
    print(text)
   

    messages = [
    {"role": "user", "content": text}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        n=1,
        stop=None
    )
    print(f"GPT Result : {response.choices[0].message['content'].strip()}")

    ## onnx 
    # torch.onnx.export(
    #     stt.asr_model.encoder, 
    #     (input_dynamic, input_static), 
    #     'model_encoder.onnx', 
    #     input_names=['input_a'], 
    #     output_names=['output'],
    #     dynamic_axes={
    #         'input_a': {1: 'N'},  # Declare axis 1 as dynamic (variable length)
    #         'output': {1: 'N'}  # Declare axis 1 as dynamic for the output as well
    #     },
    #     opset_version=11
    # )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument('--model_path', type=str, required=True, help='model path')
    parser.add_argument('--model_config_path', type=str, required=True, help='model config path')
    parser.add_argument('--inference_config', type=str, required=True, help='inference config')
    parser.add_argument('--device', type=str, required=True, help='device')
    parser.add_argument('--ctc_weight', type=float, required=True, help='ctc_weight')
    parser.add_argument('--beam_size', type=int, required=True, help='beam search size')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--nbest', type=int, required=True, help='Nbest')


    args = parser.parse_args()
    main(args)
