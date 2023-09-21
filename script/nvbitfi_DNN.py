import numpy as np
import copy
import torch
import torch.nn as nn
import os, sys
import h5py
import json

DEBUG = 0

torch.backends.cudnn.enabled = True
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False
# print(self.Input_dataset[0:128].shape)


class extract_embeddings_nvbit:
    def __init__(self, model, lyr_type=[nn.Conv2d], lyr_num=[0], batch_size=1) -> None:
        self.DNN = model
        self.batch_size = batch_size
        self.layer_id = 0
        self.extracted_layer = {}
        self.layer_types = lyr_type
        self.layer_number = lyr_num
        self.Input_dataset = None
        self.Golden_dataset = None
        self.Results_dataset = None
        self.DNN_targets = []
        self.DNN_outputs = []

        self.layer_embedding_list_input = {}
        self.layer_embedding_list_output = {}
        self.handles, self.layer_model = self._traverse_model_set_hooks_neurons(
            self.DNN
        )
        print(self.layer_id, self.layer_number)

    def _get_layer_embeddings(self, name):
        dtypes = [torch.quint8, torch.uint8, torch.qint8, torch.int8]

        def hook(_, input, output):
            if name not in self.layer_embedding_list_input:
                self.layer_embedding_list_input[name] = []
                self.layer_embedding_list_output[name] = []
            if input[0].dtype in dtypes:
                self.layer_embedding_list_input[name].append(
                    copy.deepcopy(input[0].int_repr())
                )
            else:
                self.layer_embedding_list_input[name].append(copy.deepcopy(input[0]))
            if output.dtype in dtypes:
                self.layer_embedding_list_output[name].append(
                    copy.deepcopy(output.int_repr().detach())
                )
            else:
                self.layer_embedding_list_output[name].append(
                    copy.deepcopy(output.detach())
                )

        return hook

    def _traverse_model_set_hooks_neurons(self, model):
        handles = []
        for name, layer in model.named_children():
            # leaf node
            if list(layer.children()) == []:
                if "all" in self.layer_types:
                    if (self.layer_id in self.layer_number) or (
                        self.layer_number[0] == -1
                    ):
                        # print(self.layer_id, name,type(layer))
                        self.extracted_layer[
                            f"layer_id_{self.layer_id}"
                        ] = copy.deepcopy(layer)
                        hook = self._get_layer_embeddings(f"layer_id_{self.layer_id}")
                        handles.append(layer.register_forward_hook(hook))
                    self.layer_id += 1
                else:
                    for i in self.layer_types:
                        if isinstance(layer, i):
                            if (self.layer_id in self.layer_number) or (
                                self.layer_number[0] == -1
                            ):
                                # print(self.layer_id, name,type(layer))
                                self.extracted_layer[
                                    f"layer_id_{self.layer_id}"
                                ] = copy.deepcopy(layer)
                                hook = self._get_layer_embeddings(
                                    f"layer_id_{self.layer_id}"
                                )
                                handles.append(layer.register_forward_hook(hook))
                            self.layer_id += 1
            # unpack node
            else:
                subHandles = self._traverse_model_set_hooks_neurons(layer)
                for i in subHandles:
                    handles.append(i)
        return handles, self.extracted_layer

    def DNN_inference(self, input, targets):
        Outputs = self.DNN(input)
        self.DNN_targets.append(targets)
        self.DNN_outputs.append(Outputs)
        return Outputs

    def DNN_run_embeddings_model(
        self, loader, device=torch.device("cpu:0"), max_images=-1
    ):
        self.DNN.to(device)
        self.DNN.eval()
        running_corrects = 0
        for idx, (inputs, labels) in enumerate(loader):
            # print(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = self.DNN(inputs)
            self.DNN_targets.append(labels)
            self.DNN_outputs.append(outputs)

            _, preds = torch.max(outputs, 1)

            # statistics
            running_corrects += torch.sum(preds == labels.data)
            eval_accuracy = running_corrects / (idx * len(labels) + len(labels))

            if (max_images > 0) and ((idx * len(labels) + len(labels)) >= max_images):
                print(f"accuracy = {100*eval_accuracy}%")
                break

        print(f"accuracy = {100*eval_accuracy}%")

    def _save_target_layer(
        self, state, checkpoint="./checkpoint", filename="checkpoint.pth.tar"
    ):
        os.system(f"mkdir -p {checkpoint}")
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)

    def extract_embeddings_target_layer(self):
        for layer_id in self.layer_embedding_list_input:
            if DEBUG:
                print(len(self.layer_embedding_list_input[layer_id]))
            if DEBUG:
                print(len(self.layer_embedding_list_output[layer_id]))
            if DEBUG:
                print(self.layer_model[layer_id])

            if DEBUG:
                print((self.layer_embedding_list_input[layer_id][0].shape))
            if DEBUG:
                print((self.layer_embedding_list_output[layer_id][0].shape))

            if DEBUG:
                print((self.layer_embedding_list_input[layer_id]))
            if DEBUG:
                print((self.layer_embedding_list_output[layer_id]))

            current_path = os.path.dirname(__file__)
            embeddings_input = (
                torch.cat(self.layer_embedding_list_input[layer_id]).cpu().numpy()
            )
            embeddings_output = (
                torch.cat(self.layer_embedding_list_output[layer_id]).cpu().numpy()
            )
            DNN_targets = torch.cat(self.DNN_targets).cpu().numpy()

            DNN_Outputs = torch.cat(self.DNN_outputs).cpu().numpy()

            log_path_file = os.path.join(current_path, f"embeddings_{layer_id}.h5")

            with h5py.File(log_path_file, "w") as hf:
                hf.create_dataset(
                    "layer_input", data=embeddings_input, compression="gzip"
                )
                hf.create_dataset(
                    "layer_output", data=embeddings_output, compression="gzip"
                )
                hf.create_dataset("DNN_targets", data=DNN_targets, compression="gzip")
                hf.create_dataset("DNN_outputs", data=DNN_Outputs, compression="gzip")
                hf.create_dataset(
                    "sample_id",
                    data=range(len(self.layer_embedding_list_input[layer_id])),
                    compression="gzip",
                )
                # hf.create_dataset('batch_size', data=self.batch_size, compression="gzip")

            layer_inputs_path_file = os.path.join(
                current_path, f"inputs_layer_{layer_id}.h5"
            )
            np.save(f"inputs_layer_{layer_id}.npy", embeddings_input)
            with h5py.File(layer_inputs_path_file, "w") as hf:
                hf.create_dataset(
                    "layer_input", data=embeddings_input, compression="gzip"
                )

            layer_inputs_path_file = os.path.join(
                current_path, f"Golden_Output_layer_{layer_id}.h5"
            )
            np.save(f"Golden_Output_layer_{layer_id}.npy", embeddings_output)
            with h5py.File(layer_inputs_path_file, "w") as hf:
                hf.create_dataset(
                    "layer_output", data=embeddings_output, compression="gzip"
                )
            # print(self.layer_model[layer_id])
            self._save_target_layer(
                self.layer_model[layer_id], filename=f"target_layer_{layer_id}.pth.tar"
            )


class load_embeddings:
    def __init__(self, layer_number, batch_size=1, layer_output_shape=(1,)) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.layer_results = []
        self.layer_number = layer_number
        current_path = os.path.dirname(__file__)
        model_file = os.path.join(current_path, "checkpoint", "target_layer.pth.tar")
        # dataset_file = os.path.join(
        #     current_path,
        #     f"embeddings_batch_size_{self.batch_size}_layer_id_{self.layer_number}.h5",
        # )
        dataset_file = os.path.join(
            current_path,
            f"inputs_layer.h5",
        )
        # self.layer_model = torch.load(model_file, map_location=torch.device("cpu"))
        self.layer_model = torch.load(model_file)
        self.layer_model = self.layer_model.to(self.device)
        self.layer_model.eval()

        if DEBUG:
            print(self.layer_model)

        with h5py.File(dataset_file, "r") as hf:
            self.Input_dataset = np.array(hf["layer_input"])
            # self.Output_dataset = np.array(hf["layer_output"])
            # self.batch_size=np.array(hf['batch_size'])

        if DEBUG:
            print(len(self.Input_dataset))
        # print(len(self.Output_dataset))

        if DEBUG:
            print((self.Input_dataset.shape))
        # print((self.Output_dataset.shape))
        # print(self.batch_size)

        self.onnx_model_name = "Layer_pytorch.onnx"
        self.TRT_model_name = "Layer_pytorch.rtr"
        self.TRT_output_shape = None

    def layer_inference(self):
        max_batches = float(float(len(self.Input_dataset)) / float(self.batch_size))
        with torch.no_grad():
            for batch in range(0, int(np.ceil(max_batches))):
                img = self.Input_dataset[
                    batch * self.batch_size : batch * self.batch_size + self.batch_size
                ]
                img_tensor = torch.from_numpy(img)
                img_tensor = img_tensor.to(self.device)
                # targets = self.Output_dataset[
                #     batch * self.batch_size : batch * self.batch_size + self.batch_size
                # ]

                out = self.layer_model(img_tensor)
                if batch == 0:
                    self.TRT_output_shape = out.shape
                """
                Golden_output = (
                    torch.from_numpy(
                        self.Output_dataset[
                            batch * self.batch_size : batch * self.batch_size
                            + self.batch_size
                        ]
                    )
                ).to(self.device)
                """
                # if not torch.equal(out, Golden_output):
                #    print("Not getting the expected result!")
                # np_out = out.cpu().detach().numpy()

                self.layer_results.append(out.cpu().detach())

                # print(img_tensor.shape)
                # print(out.shape)
                # print(Golden_output.shape)
                # print(torch.eq(out,Golden_output))
                # print(targets-np_out)
                # break
        embeddings_outputs = torch.cat(self.layer_results).numpy()

        current_path = os.path.dirname(__file__)
        log_path_file = os.path.join(
            current_path,
            f"Output_layer.h5",
        )

        with h5py.File(log_path_file, "w") as hf:
            hf.create_dataset(
                "layer_output", data=embeddings_outputs, compression="gzip"
            )

    def TRT_create_onnx_model(self):
        max_batches = float(float(len(self.Input_dataset)) / float(self.batch_size))
        batch = 0
        img = self.Input_dataset[
            batch * self.batch_size : batch * self.batch_size + self.batch_size
        ]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(self.device)
        torch.onnx.export(
            self.layer_model, img_tensor, self.onnx_model_name, verbose=False
        )
        USE_FP16 = False
        target_dtype = np.float16 if USE_FP16 else np.float32
        if USE_FP16:
            cmd = f"/usr/src/tensorrt/bin/trtexec --onnx={self.onnx_model_name} --saveEngine={self.TRT_model_name} --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16 "
        else:
            cmd = f"/usr/src/tensorrt/bin/trtexec --onnx={self.onnx_model_name} --saveEngine={self.TRT_model_name} --explicitBatch "

        msg = os.system(cmd)
        print(cmd)


class extract_statistics_nvbit:
    def __init__(self, model, lyr_type=[nn.Conv2d], lyr_num=[0], batch_size=1,valid_layers=[]) -> None:
        self.DNN = model
        self.batch_size = batch_size
        self.layer_id = 0
        self.extracted_layer = {}
        self.layer_types = lyr_type
        self.layer_number = lyr_num
        self.Input_dataset = None
        self.Golden_dataset = None
        self.Results_dataset = None
        self.valid_layers=valid_layers
        self.DNN_targets = []
        self.DNN_outputs = []

        self.layer_embedding_list_input = {}
        self.layer_embedding_list_output = {}
        self.handles, self.layer_model = self._traverse_model_set_hooks_neurons(
            self.DNN
        )
        print(self.layer_id, self.layer_number)

    def _get_layer_embeddings(self, name):
        dtypes = [torch.quint8, torch.uint8, torch.qint8, torch.int8]

        def hook(_, input, output):
            if name not in self.layer_embedding_list_input:
                self.layer_embedding_list_input[name] = []
                self.layer_embedding_list_output[name] = []
            # if DEBUG: print(input)
            input_cpu = input[0].cpu().detach()
            output_cpu = output.cpu().detach()
            self.layer_embedding_list_input[name].append(copy.deepcopy(input_cpu))
            self.layer_embedding_list_output[name].append(copy.deepcopy(output_cpu))
            # self.layer_embedding_list_output[name].append(
            #    copy.deepcopy(output.detach())
            # )
        return hook

    def _traverse_model_set_hooks_neurons(self, model):
        handles = []
        for names, layer in model.named_children():
            # leaf node
            print(layer.__class__.__name__)
            print(type(layer))
            name = f"layer-id-{self.layer_id}-{layer.__class__.__name__}" # just the name i want to add to the layer
            if list(layer.children()) == [] or (any([True for target in self.valid_layers if isinstance(layer,target)])):
                if "all" in self.layer_types:
                    if (self.layer_id in self.layer_number) or (
                        self.layer_number[0] == -1
                    ):
                        # print(self.layer_id, name,type(layer))
                        self.extracted_layer[name] = copy.deepcopy(layer)
                        hook = self._get_layer_embeddings(name)
                        handles.append(layer.register_forward_hook(hook))
                    self.layer_id += 1
                else:
                    for i in self.layer_types:
                        if isinstance(layer, i):
                            if (self.layer_id in self.layer_number) or (
                                self.layer_number[0] == -1
                            ):
                                # print(self.layer_id, name,type(layer))
                                self.extracted_layer[
                                    name
                                ] = copy.deepcopy(layer)
                                hook = self._get_layer_embeddings(name)
                                handles.append(layer.register_forward_hook(hook))
                            self.layer_id += 1
            # unpack node
            else:
                subHandles = self._traverse_model_set_hooks_neurons(layer)
                for i in subHandles:
                    handles.append(i)
        return handles, self.extracted_layer

    def DNN_inference(self, input, targets):
        Outputs = self.DNN(input)
        self.DNN_targets.append(targets)
        self.DNN_outputs.append(Outputs)
        return Outputs

    def DNN_run_embeddings_model(
        self, loader, device=torch.device("cpu:0"), max_images=-1
    ):
        self.DNN.to(device)
        self.DNN.eval()
        running_corrects = 0
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(loader):
                # print(labels)
                inputs = inputs.to(device, non_blocking=True)
                #labels = labels.to(device)
                outputs = self.DNN(inputs)
                self.DNN_targets.append(labels.cpu())
                self.DNN_outputs.append(outputs.cpu())

                _, preds = torch.max(outputs, 1)

                # statistics
                running_corrects += torch.sum(preds.cpu() == labels.data)
                eval_accuracy = running_corrects / (idx * len(labels) + len(labels))
                self.extract_embeddings_target_layer()
                if (max_images > 0) and ((idx * len(labels) + len(labels)) >= max_images):
                    print(f"{idx*len(labels)} images processed")
                    print(f"accuracy = {100*eval_accuracy}%")
                    break
                print(f"{idx*len(labels)} images processed")
                # GPUtil.showUtilization()
            print(f"accuracy = {100*eval_accuracy}%")

    def _save_target_layer(
        self, state, checkpoint="./checkpoint", filename="checkpoint.pth.tar"
    ):
        os.system(f"mkdir -p {checkpoint}")
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)

    def extract_embeddings_target_layer(self):
        for layer_id in self.layer_embedding_list_input:
            print(layer_id)
            if DEBUG:
                print(len(self.layer_embedding_list_input[layer_id]))
            if DEBUG:
                print(len(self.layer_embedding_list_output[layer_id]))
            if DEBUG:
                print(self.layer_model[layer_id])

            if DEBUG:
                print((self.layer_embedding_list_input[layer_id][0]))
            if DEBUG:
                print((self.layer_embedding_list_output[layer_id][0]))

            if DEBUG:
                print((self.layer_embedding_list_input[layer_id]))
            if DEBUG:
                print((self.layer_embedding_list_output[layer_id]))

            embeddings_input = (
                torch.cat(self.layer_embedding_list_input[layer_id]).cpu().numpy()
            )
            embeddings_output = (
                torch.cat(self.layer_embedding_list_output[layer_id]).cpu().numpy()
            )


            current_path = os.path.expanduser(os.path.dirname(__file__))
            dir1 = os.path.join(current_path,"Embeddings_inputs")
            os.system(f"mkdir -p {dir1}")

            dir2 = os.path.join(current_path,"Embeddings_outputs")
            os.system(f"mkdir -p {dir2}")

            filename1 = os.path.join(dir1,f"{layer_id}.npy")
            filename2 = os.path.join(dir2,f"{layer_id}.npy")

            if os.path.exists(filename1):
                tmpIn1 = np.load(filename1)
                tmpIn2 = np.concatenate([tmpIn1,embeddings_input])
                np.save(filename1,tmpIn2)
                tmpOut1 = np.load(filename1)
                tmpOut2 = np.concatenate([tmpOut1,embeddings_input])
                np.save(filename2,tmpOut2)
            else:
                np.save(filename1,embeddings_input)
                np.save(filename2,embeddings_output)

            """
            ins = torch.from_numpy(embeddings_input)
            out = torch.from_numpy(embeddings_output)

            in_mean = torch.mean(ins)
            in_median = torch.median(ins)
            in_std = torch.std(ins)
            in_max = torch.max(ins)
            in_min = torch.min(ins)
            in_var = torch.var(ins)
            in_nz = torch.count_nonzero(ins)
            in_tot = len(ins.reshape(ins.size(0), -1))*len(ins.reshape(ins.size(0), -1)[0])
            in_hist = torch.histogram(ins, bins=100, density=True)
            in_dict = {
                "mean": in_mean.detach().numpy().tolist(),
                "median": in_median.detach().numpy().tolist(),
                "std": in_std.detach().numpy().tolist(),
                "max": in_max.detach().numpy().tolist(),
                "min": in_min.detach().numpy().tolist(),
                "var": in_var.detach().numpy().tolist(),
                "nz": in_nz.detach().numpy().tolist(),
                "tot": in_tot,
                "hist": {
                    "hist_distr": in_hist.hist.detach().numpy().tolist(),
                    "bin_edges": in_hist.bin_edges.detach().numpy().tolist(),
                },
            }
            
            out_mean = torch.mean(out)
            out_median = torch.median(out)
            out_std = torch.std(out)
            out_max = torch.max(out)
            out_min = torch.min(out)
            out_var = torch.var(out)
            out_nz = torch.count_nonzero(out)
            out_tot = len(out.reshape(out.size(0), -1))*len(out.reshape(out.size(0), -1)[0])
            out_hist = torch.histogram(out, bins=100, density=True)
            out_dict = {
                "mean": out_mean.detach().numpy().tolist(),
                "median": out_median.detach().numpy().tolist(),
                "std": out_std.detach().numpy().tolist(),
                "max": out_max.detach().numpy().tolist(),
                "min": out_min.detach().numpy().tolist(),
                "var": out_var.detach().numpy().tolist(),
                "nz": out_nz.detach().numpy().tolist(),
                "tot": out_tot,
                "hist": {
                    "hist_distr": out_hist.hist.detach().numpy().tolist(),
                    "bin_edges": out_hist.bin_edges.detach().numpy().tolist(),
                },
            }


            with open(os.path.join(dir1,f"{layer_id}.json"), "w") as outfile:
                json.dump(in_dict, outfile)

            with open(os.path.join(dir2,f"{layer_id}.json"), "w") as ooutfile:
                json.dump(out_dict, ooutfile)
            """





class Extract_gradients_layers:
    def __init__(self, model, lyr_type=[nn.Conv2d], lyr_num=[0], batch_size=1) -> None:
        self.DNN = model
        self.batch_size = batch_size
        self.layer_id = 0
        self.extracted_layer = {}
        self.layer_types = lyr_type
        self.layer_number = lyr_num
        self.Input_dataset = None
        self.Golden_dataset = None
        self.Results_dataset = None
        self.DNN_targets = []
        self.DNN_outputs = []

        self.layer_embedding_list_input = {}
        self.layer_embedding_list_output = {}
        self.handles, self.layer_model = self._traverse_model_set_hooks_neurons(
            self.DNN
        )
        print(self.layer_id, self.layer_number)

    def _get_layer_embeddings(self, name):
        dtypes = [torch.quint8, torch.uint8, torch.qint8, torch.int8]
        def hook(module, grad_input, grad_output):
            if name not in self.layer_embedding_list_input:
                self.layer_embedding_list_input[name] = []
                self.layer_embedding_list_output[name] = []
            # if DEBUG: print(input)
            input_cpu = grad_input[0].cpu().detach()
            output_cpu = grad_output[0].cpu().detach()
            self.layer_embedding_list_input[name].append(copy.deepcopy(input_cpu))
            self.layer_embedding_list_output[name].append(copy.deepcopy(output_cpu))
            # self.layer_embedding_list_output[name].append(
            #    copy.deepcopy(output.detach())
            # )
        return hook
    
    # Due to possible issues when usinf ReLu implace the gradients are calculated
    # by using the trick presented here: https://github.com/pytorch/pytorch/issues/61519
    # using the forward hook to get the tesnor, then registering a hook on each tensor to capture the 
    # gradients

    def _tensor_hook_inputs(self,name):
        def _tensor_hook(grad): 
            if name not in self.layer_embedding_list_input:
                self.layer_embedding_list_input[name] = []
            input_cpu = grad.cpu().detach()
            self.layer_embedding_list_input[name].append(copy.deepcopy(input_cpu))         
        return _tensor_hook



    def _tensor_hook_outputs(self,name):
        def _tensor_hook(grad): 
            if name not in self.layer_embedding_list_output:
                self.layer_embedding_list_output[name] = []
            output_cpu = grad.cpu().detach()
            self.layer_embedding_list_output[name].append(copy.deepcopy(output_cpu))         
        return _tensor_hook



    def _fw_wrapper_hook(self,name):
        def _fw_hook(mod, input, output):
            input[0].register_hook(self._tensor_hook_inputs(name))
            output.register_hook(self._tensor_hook_outputs(name))
        return _fw_hook

    def _traverse_model_set_hooks_neurons(self, model):
        handles = []
        for names, layer in model.named_children():
            # leaf node
            print(layer.__class__.__name__)
            name = f"layer-id-{self.layer_id}-{layer.__class__.__name__}" # just the name i want to add to the layer
            if list(layer.children()) == []:
                if "all" in self.layer_types:
                    if (self.layer_id in self.layer_number) or (
                        self.layer_number[0] == -1
                    ):
                        # print(self.layer_id, name,type(layer))
                        self.extracted_layer[name] = copy.deepcopy(layer)
                        #hook = self._get_layer_embeddings(name)
                        hook = self._fw_wrapper_hook(name)
                        # handles.append(layer.register_full_backward_hook(hook))
                        handles.append(layer.register_forward_hook(hook))
                    self.layer_id += 1
                else:
                    for i in self.layer_types:
                        if isinstance(layer, i):
                            if (self.layer_id in self.layer_number) or (
                                self.layer_number[0] == -1
                            ):
                                # print(self.layer_id, name,type(layer))
                                self.extracted_layer[name] = copy.deepcopy(layer)
                                #hook = self._get_layer_embeddings(name)
                                hook = self._fw_wrapper_hook(name)
                                # handles.append(layer.register_full_backward_hook(hook))
                                handles.append(layer.register_forward_hook(hook))
                            self.layer_id += 1
            # unpack node
            else:
                subHandles = self._traverse_model_set_hooks_neurons(layer)
                for i in subHandles:
                    handles.append(i)
        return handles, self.extracted_layer

    def DNN_inference(self, input, targets):
        Outputs = self.DNN(input)
        self.DNN_targets.append(targets)
        self.DNN_outputs.append(Outputs)
        return Outputs

    def DNN_run_embeddings_model(
        self, loader, device=torch.device("cpu:0"), max_images=-1, target_label = -1
    ):        
        self.DNN.to(device)
        for param in self.DNN.parameters():
            param.requires_grad = False

        self.DNN.eval()
        counter=0
        for idx, (inputs, labels) in enumerate(loader):
            # print(labels)
            if target_label!= -1:
                if labels == target_label:
                    inputs = inputs.to(device, non_blocking=True)
                    inputs.requires_grad = True
                    labels = labels.to(device)
                    outputs = self.DNN(inputs)
                    score, preds = torch.max(outputs, 1)
                    if preds == labels: 
                        score.backward()
                        counter+=1
            else:
                inputs = inputs.to(device, non_blocking=True)
                inputs.requires_grad = True
                labels = labels.to(device)
                outputs = self.DNN(inputs)
                score, preds = torch.max(outputs, 1)
                if preds == labels: 
                    score.backward()
                    counter+=1

            if (max_images > 0) and ((counter * len(labels) + len(labels)) > max_images):
                break
            

    def _save_target_layer(
        self, state, checkpoint="./checkpoint", filename="checkpoint.pth.tar"
    ):
        os.system(f"mkdir -p {checkpoint}")
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)

    def extract_embeddings_target_layer(self):
        import matplotlib.pyplot as plt
        for layer_id in self.layer_embedding_list_input:
            print(layer_id)
            if DEBUG:
                print(len(self.layer_embedding_list_input[layer_id]))
            if DEBUG:
                print(len(self.layer_embedding_list_output[layer_id]))
            if DEBUG:
                print(self.layer_model[layer_id])

            if DEBUG:
                print((self.layer_embedding_list_input[layer_id][0]))
            if DEBUG:
                print((self.layer_embedding_list_output[layer_id][0]))

            if DEBUG:
                print((self.layer_embedding_list_input[layer_id]))
            if DEBUG:
                print((self.layer_embedding_list_output[layer_id]))


            grad = torch.cat(self.layer_embedding_list_output[layer_id])

            #for grad in self.layer_embedding_list_output[layer_id]:

            #grad_input = self.layer_embedding_list_output[layer_id][0].cpu()
            grad_input = grad.cpu()
            saliency = torch.sum(torch.abs(grad_input), dim=0)/len(grad_input)
            
            #saliency = saliency.apply_(lambda x: 0 if x<0 else x)
            print(grad_input.size())
            print(saliency.size())
            
            dims = grad_input.size()
            """
            for b in range(len(grad_input)):
                bi = grad_input[b]
                if len(dims)>3:
                    summ = torch.zeros(bi[0].shape)
                    for c in range(len(bi)):
                        #summ += torch.abs(bi[c])
                        im = torch.abs(bi[c])
                        im = im.apply_(lambda x: 0 if x<0.4 else x)
                        s = (im - im.min())/(im.max()-im.min())
                        print(s.shape)
                        print(bi.shape)
                        plt.imshow(s,cmap=plt.cm.hot)
                        plt.axis('off')
                        plt.show()

                else:
                    summ = grad_input
                    s = (summ - summ.min())/(summ.max()-summ.min())
                    print(s.shape)
                    print(bi.shape)
                    plt.imshow(s,cmap=plt.cm.hot)
                    plt.axis('off')
                    plt.show()
            """

            if len(saliency.size())>2:
                for ch in range(len(saliency)):
                    # saliency[ch] = saliency[ch].apply_(lambda x: 0 if x<0 else x)
                    saliencyx = (saliency[ch] - saliency[ch].min())/(saliency[ch].max()-saliency[ch].min())
                    print(saliency[ch].max(),saliency[ch].min(), saliency[ch].mean(), saliency[ch].median())
                    plt.imshow(saliencyx,cmap=plt.cm.hot)
                    plt.axis('off')
                    plt.show()
            else:                
                saliency = saliency.reshape(1,-1)
                saliencyx = (saliency - saliency.min())/(saliency.max()-saliency.min())
                print(saliency.max(),saliency.min(), saliency.mean(), saliency.median())
                plt.imshow(saliencyx,cmap=plt.cm.hot)
                plt.axis('off')
                plt.show()
                print(saliency)
                print(saliencyx)



def analize_numpy_embeddings():
    import pandas as pd
    current_path = os.path.expanduser(os.path.dirname(__file__))
    df = pd.DataFrame()
    BINS = 50

    dir1 = os.path.join(current_path,"Embeddings_inputs")
    dir2 = os.path.join(current_path,"Embeddings_outputs")

    numpy_files_1 = [dirname for dirname in os.listdir(dir1) if ".npy" in dirname]
    numpy_files_2 = [dirname for dirname in os.listdir(dir2) if ".npy" in dirname ]
    print(numpy_files_1)
    print(numpy_files_2)

    os.chdir(dir1)
    for layer_id in range(len(numpy_files_1)):
        filename = next(x for x in numpy_files_1 if f"layer-id-{layer_id}-" in x)
        if filename != "":
            embeddings_input = np.load(filename)
            ins = torch.from_numpy(embeddings_input)
            in_mean = torch.mean(ins)
            in_median = torch.median(ins)
            in_std = torch.std(ins)
            in_max = torch.max(ins)
            in_min = torch.min(ins)
            in_var = torch.var(ins)
            in_nz = torch.count_nonzero(ins)
            in_tot = len(ins.reshape(ins.size(0), -1))*len(ins.reshape(ins.size(0), -1)[0])
            in_hist = torch.histogram(ins, bins=BINS, density=True)
            in_dict = {
                "mean": in_mean.detach().numpy().tolist(),
                "median": in_median.detach().numpy().tolist(),
                "std": in_std.detach().numpy().tolist(),
                "max": in_max.detach().numpy().tolist(),
                "min": in_min.detach().numpy().tolist(),
                "var": in_var.detach().numpy().tolist(),
                "nz": in_nz.detach().numpy().tolist(),
                "tot": in_tot,
                "hist": {
                    "hist_distr": in_hist.hist.detach().numpy().tolist(),
                    "bin_edges": in_hist.bin_edges.detach().numpy().tolist(),
                    "hist_prob": np.sum(in_hist.hist.detach().numpy()).tolist()
                },
            }

            json_filename = filename.replace(".npy", ".json")
            with open(os.path.join(dir1,json_filename), "w") as outfile:
                json.dump(in_dict, outfile)

            print(os.path.join(dir1,json_filename))
            df1 = pd.DataFrame({
                "layerID": filename,
                "mean": in_mean.detach().numpy().tolist(),
                "median": in_median.detach().numpy().tolist(),
                "std": in_std.detach().numpy().tolist(),
                "max": in_max.detach().numpy().tolist(),
                "min": in_min.detach().numpy().tolist(),
                "var": in_var.detach().numpy().tolist(),
                "nz": in_nz.detach().numpy().tolist(),
                "tot": in_tot
            },index=[0])

            df = pd.concat([df, df1], ignore_index=True)

    df.to_csv(os.path.join(dir1,"layer-input-distributions.csv"))


        # fig = plt.figure()
        # x = np.array(in_dict["hist"]["bin_edges"][2:])
        # y = np.array(in_dict["hist"]["hist_distr"][1:])
        # plt.stem(x,y)
        # plt.xlabel('Plot Number')
        # plt.ylabel('Important var')
        # plt.title('Interesting Graph\nCheck it out')
        # plt.show()
        
    df = pd.DataFrame()
    os.chdir(dir2)
    for layer_id in range(len(numpy_files_2)):
        filename = next(x for x in numpy_files_2 if f"layer-id-{layer_id}-" in x)
        if filename != "":
            embeddings_output = np.load(filename)
            out = torch.from_numpy(embeddings_output)            
            out_mean = torch.mean(out)
            out_median = torch.median(out)
            out_std = torch.std(out)
            out_max = torch.max(out)
            out_min = torch.min(out)
            out_var = torch.var(out)
            out_nz = torch.count_nonzero(out)
            out_tot = len(out.reshape(out.size(0), -1))*len(out.reshape(out.size(0), -1)[0])
            out_hist = torch.histogram(out, bins=BINS, density=True)
            out_dict = {
                "mean": out_mean.detach().numpy().tolist(),
                "median": out_median.detach().numpy().tolist(),
                "std": out_std.detach().numpy().tolist(),
                "max": out_max.detach().numpy().tolist(),
                "min": out_min.detach().numpy().tolist(),
                "var": out_var.detach().numpy().tolist(),
                "nz": out_nz.detach().numpy().tolist(),
                "tot": out_tot,
                "hist": {
                    "hist_distr": out_hist.hist.detach().numpy().tolist(),
                    "bin_edges": out_hist.bin_edges.detach().numpy().tolist(),
                    "hist_prob": np.sum(out_hist.hist.detach().numpy()).tolist()
                },
            }

            json_filename = filename.replace(".npy", ".json")
            with open(os.path.join(dir2,json_filename), "w") as outfile:
                json.dump(out_dict, outfile)

            print(os.path.join(dir2,json_filename))
            # fig = plt.figure()
            # x = np.array(out_dict["hist"]["bin_edges"][1:])
            # y = np.array(out_dict["hist"]["hist_distr"])
            # plt.stem(x,y)
            # plt.xlabel('Plot Number')
            # plt.ylabel('Important var')
            # plt.title('Interesting Graph\nCheck it out')
            # plt.show()
            df1 = pd.DataFrame({
                "layerID": filename,
                "mean": out_mean.detach().numpy().tolist(),
                "median": out_median.detach().numpy().tolist(),
                "std": out_std.detach().numpy().tolist(),
                "max": out_max.detach().numpy().tolist(),
                "min": out_min.detach().numpy().tolist(),
                "var": out_var.detach().numpy().tolist(),
                "nz": out_nz.detach().numpy().tolist(),
                "tot": out_tot
            },index=[0])

            df = pd.concat([df, df1], ignore_index=True)

    df.to_csv(os.path.join(dir2,"layer-output-distributions.csv"))