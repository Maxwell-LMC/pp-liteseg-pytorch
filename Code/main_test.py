import pytorch_liteseg
from STDC1_pytorch_raw.model import STDC1
import torch
import torch.onnx

###initialise liteseg model
model = pytorch_liteseg.liteseg(num_classes=10, encoder=STDC1)

feature = torch.randn((1,3,512,1024))
model.eval()
with torch.inference_mode():
    output = model(feature)
    print(output.shape)


#Function to Convert to ONNX 
def Convert_ONNX(): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn((1,3,512,1024), requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "pytorch-liteseg.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

Convert_ONNX()