import torch
import onnx
from onnxsim import simplify

from darknetonnx.darknet import Darknet


def transform_to_onnx(cfgfile, weightfile, outputfile, batch_size=1):
    # Load model
    model = Darknet(cfgfile)
    model.load_weights(weightfile)
    model.eval()
    # Prepare ONNX config
    input_names = ["input"]
    output_names = ["output"]
    dynamic = False
    dynamic_axes = None
    if batch_size < 0:
        dynamic = True
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    b = -1 if dynamic else batch_size
    if b < 0:
        b = 1
    x = torch.randn((b, 3, model.height, model.width))
    # Export the model
    torch.onnx.export(
        model,
        x,
        outputfile,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    # ONNX simplifier
    onnx_model = onnx.load(outputfile)
    if dynamic:
        input_shapes = {input_names[0]: list(x.shape)}
    model_simp, check = simplify(
        onnx_model, dynamic_input_shape=dynamic, input_shapes=input_shapes
    )
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, outputfile)
    return outputfile
