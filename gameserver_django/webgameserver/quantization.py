"""
author: Tim Straube
contact: hi@optimalpi.com
licence: MIT
"""

import torch
import onnx
import tensorflow
# import onnx_tf
import random
import numpy
import onnxruntime as ort
import os
from onnxruntime.quantization import quantize_dynamic
from onnxruntime.quantization import QuantType
from webgameserver.residualnetwork import ResidualNetwork
from webgameserver.game import Battleship
from webgameserver.mcts import MCTS
        
class ConvertModel():
    def __init__(self):
        self.board_size = 9
        self.current_model_folder = "zeus9x9"
        self.game = Battleship(self.board_size)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.model = ResidualNetwork(
            self.game, 
            8, 
            8, 
            4, 
            device = self.device
        )
        self.model.load_state_dict(
            torch.load(
                f'./models/{self.current_model_folder}/main.pt', 
                map_location = self.device
            )
        )
        # Ensure the model is on the correct device
        self.model = self.model.to(self.device)  
        self.model.eval()

    def pytorch2coral(self):
        model_prefix = 'vivalavida'
        onnx_save_file = f"./models/{self.current_model_folder}/model_quantized.onnx"
        tflite_save_file = model_prefix + '.tflite'
        tflite_quant_save_file = model_prefix + '_quant.tflite'

        print('Loading ONNX and checking...')
        onnx_model = onnx.load(onnx_save_file)
        onnx.checker.check_model(onnx_model)
        print(onnx.helper.printable_graph(onnx_model.graph))

        print('Converting ONNX to TF...')
        tf_rep = onnx_tf.backend.prepare(onnx_model)
        tf_rep.export_graph(model_prefix)

        print('Converting TF to TFLite...')
        converter = tensorflow.lite.TFLiteConverter.from_saved_model(model_prefix)
        tflite_model = converter.convert()
        with open(tflite_save_file, 'wb') as f:
            f.write(tflite_model)

        print('Converting TF to Quantised TFLite...')
        state = numpy.zeros(
            (6, 9, 9), dtype=numpy.int8
        )
        def representative_data_gen():
            for i in range(100000):
                yield [
                    self.game.get_encoded_state(state).reshape(1, -1)
                ]

        converter_quant = tensorflow.lite.TFLiteConverter.from_saved_model(model_prefix)
        converter_quant.optimizations = [
            tensorflow.lite.Optimize.DEFAULT
        ]
        converter_quant.representative_dataset = representative_data_gen
        converter_quant.target_spec.supported_ops = [
            tensorflow.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tensorflow.lite.OpsSet.SELECT_TF_OPS
        ]
        converter_quant.target_spec.supported_types = [
            tensorflow.int8
        ]
        # Just accept that observations and actions are inherently floaty, let Coral handle that on the CPU
        converter_quant.inference_input_type = tensorflow.float32
        converter_quant.inference_output_type = tensorflow.float32
        tflite_quant_model = converter_quant.convert()
        with open(tflite_quant_save_file, 'wb') as f:
            f.write(tflite_quant_model)

        print('Converting TFLite [nonquant] to Coral...')
        system('edgetpu_compiler --show_operations -o ' + os.path.dirname(model_prefix) + ' ' + tflite_save_file)

        print('Converting TFLite [quant] to Coral...')
        system(
            'edgetpu_compiler --show_operations -o ' + os.path.dirname(model_prefix) + ' ' + tflite_quant_save_file
        )

    def pytorch2tflite(self):
        state = self.game.restart(0)

        random_number_moves = random.randint(
            28, 
            2 * self.board_size * self.board_size
        )
        args = {
            'C': 2,
            'num_searches': 100,
            'dirichlet_epsilon': 0,
            'dirichlet_alpha': 0.1
        }
        mcts = MCTS(self.game, args, self.model)
        player = -1

        for _ in range(random_number_moves):
            neutral_state = self.game.change_perspective(
                self.game.state.copy(), player
            )
            mcts_probs = mcts.search(neutral_state)
            action = numpy.argmax(mcts_probs)
            state = self.game.step(
                self.game.state, 
                action, 
                player
            )
            player = -player
            yield state

        encoded_state = self.game.get_encoded_state(state)
        tensor_state = torch.tensor(
            encoded_state, 
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        # Convert PyTorch model to ONNX
        onnx_path = 'resnet_model.onnx'
        self.torch_to_onnx(
            self.model, 
            tensor_state, 
            onnx_path
        )
        print(f"ONNX model saved to {onnx_path}")

        # Convert ONNX model to TensorFlow
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        tf_rep = onnx_tf.backend.prepare(onnx_model)
        tf_rep.export_graph('resnet_model.pb')
        print("TensorFlow model saved to resnet_model.pb")

        # Convert TensorFlow model to TFLite
        # Load the TensorFlow model
        converter = tensorflow.lite.TFLiteConverter.from_saved_model(
            'resnet_model.pb'
        )
        tflite_model = converter.convert()

        # Save the TFLite model
        with open('resnet_model.tflite', 'wb') as file:
            file.write(tflite_model)
            
    def convert_to_onnx(self):
        onnx_path = f"./models/{self.current_model_folder}/model.onnx"
        
        # Create a dummy input tensor
        dummy_input = torch.randn(1, 4, self.board_size, self.board_size, device=self.device)
        
        # Export the model
        torch.onnx.export(
            self.model,  # model being run
            dummy_input, # model input (or a tuple for multiple inputs)
            onnx_path,                # where to save the model
            export_params=True,       # store the trained parameter weights inside the model file
            opset_version=11,         # the ONNX version to export the model to
            do_constant_folding=True, # whether to execute constant folding for optimization
            input_names = ['input'],  # the model's input names
            output_names = ['policy', 'value'], # the model's output names
            dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
            'policy' : {0 : 'batch_size'},
            'value' : {0 : 'batch_size'}}
        )
        
        print(f"Model exported to {onnx_path}")
        
        # Verify the model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid")

    def quantize_onnx_model(self):
        model_path = "./models/" + str(self.current_model_folder) + "/model.onnx"
        quantized_model_path = model_path.replace('.onnx', '_quantized.onnx')
        
        quantize_dynamic(model_path,
                        quantized_model_path,
                        weight_type=QuantType.QUInt8)
        
        print(f"Quantized model saved to: {quantized_model_path}")
        
        sess = ort.InferenceSession(quantized_model_path)

        input_name = sess.get_inputs()[0].name
        print("input name", input_name)
        input_shape = sess.get_inputs()[0].shape
        print("input shape", input_shape)
        input_type = sess.get_inputs()[0].type
        print("input type", input_type)

        dummy_input = numpy.random.randn(
            1, 
            4, 
            self.board_size, 
            self.board_size
        ).astype(numpy.float32)
        output = sess.run(None, {input_name: dummy_input})
        print(output)
        print("Quantized model inference successful")

    def torch_to_onnx(self):
        onnx_path = "./test.onnx"
        input_tensor = torch.randn(1, 4, 9, 9)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

            print("Output")

            torch.onnx.export(
                self.model, 
                input_tensor, 
                onnx_path, 
                input_names = ['input'], 
                output_names = ['output'],
                dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                opset_version = 11
            )  
            
    def static_quantization(self):
        pass

    def dynamic_quantization(self):
        model_quantized_dynamic = torch.ao.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )

        # # Copy model to qunatize
        # model_to_quantize = copy.deepcopy(self.model).to(self.device)
        # model_to_quantize.eval()
        # qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.default_dynamic_qconfig)

        # example_inputs = torch.randn(32, 4, 3, 3)

        # # prepare
        # model_prepared = quantize_fx.prepare_fx(
        #     model_to_quantize, 
        #     qconfig_mapping, 
        #     example_inputs
        # )
        # # no calibration needed when we only have dynamic/weight_only quantization
        # # quantize
        # model_quantized_dynamic = quantize_fx.convert_fx(model_prepared)

        #         # Copy model to qunatize
        # model_to_quantize = copy.deepcopy(self.model).to(self.device)
        # model_to_quantize.eval()
        # qconfig_mapping = QConfigMapping().set_global(
        #     torch.ao.quantization.default_dynamic_qconfig
        # )

        # Save the quantized model
        torch.save(
            model_quantized_dynamic.state_dict(), 
            "./models_int8/" + str(self.current_model_folder) + "/quantized_model.pth"
        )

        # If you want to save the entire model (architecture + weights)
        torch.save(
            model_quantized_dynamic, 
            "./models_int8/" + str(self.current_model_folder) + "/quantized_model_full.pth"
        )

    def test_int8_models(self):
        # warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

        loaded_model = torch.load(
            "./models_int8/" + str(self.current_model_folder) + "/quantized_model_full.pth",
            map_location=self.device
        )

        loaded_model.eval()

if __name__ == "__main__":
    convertModel = ConvertModel()
    # convertModel.pytorch2coral()
    # convertModel.torch_to_onnx()
    convertModel.convert_to_onnx()
    convertModel.quantize_onnx_model()
    # convertModel.dynamic_quantization()
    # convertModel.test_int8_models()