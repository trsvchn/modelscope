"""Model Summary.
"""
from copy import copy
from inspect import getmembers, isfunction, isbuiltin, ismethoddescriptor, ismethod
from contextlib import contextmanager
from functools import wraps
from typing import Tuple, List, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional
from torch.utils.hooks import RemovableHandle

from .core import Handle, get_size, module_walker
from .utils import size_to_str, adjust_module_name


class Summary:

    f_ignore = [
        "adaptive_avg_pool1d",
        "adaptive_max_pool1d",
        "alpha_dropout",
        "avg_pool1d",
        "batch_norm",
        "bilinear",
        "binary_cross_entropy_with_logits",
        "celu",
        "celu_",
        "channel_shuffle",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_tbc",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "cosine_embedding_loss",
        "cosine_similarity",
        "ctc_loss",
        "dropout",
        "embedding",
        "embedding_bag",
        "feature_alpha_dropout",
        "group_norm",
        "hardshrink",
        "hinge_embedding_loss",
        "instance_norm",
        "kl_div",
        "layer_norm",
        "log_softmax",
        "margin_ranking_loss",
        "max_pool1d",
        "max_pool1d_with_indices",
        "max_pool2d",
        "max_pool3d",
        "pairwise_distance",
        "pdist",
        "pixel_shuffle",
        "poisson_nll_loss",
        "prelu",
        "relu",
        "relu_",
        "rrelu",
        "rrelu_",
        "selu",
        "selu_",
        "sigmoid",
        "softmax",
        "tanh",
        "threshold",
        "threshold_",
        "triplet_margin_loss",
    ]

    fn_types = {
        "method_descriptor": "method",
        "method": "method",
        "builtin_function_or_method": "function",
        "function": "function",
    }

    def __init__(
            self,
            model,
            live: bool = True,
            model_name: Optional[str] = None,
            full_names: bool = True,
            full_type_names: bool = False,
            hide_names: Optional[List[str]] = None,
            hide_types: Optional[List[str]] = None,
            exclude_hidden: bool = True,
            fold_nodes: Optional[List[str]] = None,
            top_level: bool = False,
            low_level: bool = False,
            max_depth: int = 1000,
            col_widths: Tuple[int, int, int, int, int, int] = (5, 25, 25, 25, 15, 15),
    ):
        self.live = live
        self.model_name = model_name or model._get_name()
        self.full_names = full_names
        self.full_type_names = full_type_names
        self.hide_names = hide_names or []
        self.hide_types = hide_types or []
        self.hide_types = [t.lower() for t in self.hide_types]
        self.exclude_hidden = exclude_hidden
        self.fold_nodes = fold_nodes or []
        self.top_level = top_level
        self.low_level = low_level
        self.max_depth = max_depth
        self.col_widths = col_widths
        self.col1_w, self.col2_w, self.col3_w, self.col4_w, self.col5_w, self.col6_w = self.col_widths

        self.model = model
        self.model_backup = {}
        self.modules = module_walker((self.model_name, model), yield_parents=True)
        self.torch_module = torch
        self.torch_backup = {}

        self.tensor_module = torch.Tensor
        self.tensor_backup = {}

        self.f_module = torch.nn.functional
        self.f_backup = {}

        self.handles_pre_forward = {}
        self.handles_forward = {}

        self.depth = -1
        self.count = 1
        self.state = 1
        self.curr_module = [""]
        self.curr_parent = [""]

        self.param_ids = set()
        self.total_num_train_params = 0
        self.total_num_non_train_params = 0

        self.logs = []

        self.fold = False
        self.force_hide = False
        self.prev_is_suppressed = False

        for member in getmembers(self.torch_module):
            if isfunction(member[-1]) or isbuiltin(member[-1]):
                if not member[0].startswith("_"):
                    self.torch_backup.update({member[0]: member[-1]})

        for member in getmembers(self.tensor_module):
            if isfunction(member[-1]) or ismethoddescriptor(member[-1]):
                if not member[0].startswith("_"):
                    self.tensor_backup.update({member[0]: member[-1]})

        for member in getmembers(self.f_module):
            if isfunction(member[-1]) or isbuiltin(member[-1]):
                if (not member[0].startswith("_")) and (member[0] not in self.f_ignore):
                    self.f_backup.update({member[0]: member[-1]})

        for member in getmembers(self.model):
            if isfunction(member[-1]) or ismethod(member[-1]) or isbuiltin(member[-1]) or ismethoddescriptor(member[-1]):
                if not member[0].startswith("_"):
                    if member[0] not in [i[0] for i in getmembers(torch.nn.Module())]:
                        self.model_backup.update({member[0]: member[-1]})

    def __enter__(self):
        self.prepare_handles()
        self.register_pre_forward_hooks()
        self.register_forward_hooks()

        if self.live:
            header = f"Model: {self.model_name}\n\n" \
                     f"{'':<{self.col1_w}}Node" \
                     f"{'':<{sum(self.col_widths[1:3]) - 4}}{'Output Size':<{self.col4_w}}Params" \
                     f"{'':<{sum(self.col_widths[4:6]) - 6}}\n" \
                     f"{'':<{self.col1_w}}" \
                     f"{'Name':<{self.col2_w}}" \
                     f"{'Type':<{self.col3_w}}" \
                     f"{'':<{self.col4_w}}" \
                     f"{'Train':<{self.col5_w}}" \
                     f"{'Non-Train':<{self.col6_w}}"
            print(header)

        for k, v in self.torch_backup.items():
            setattr(self.torch_module, k, self.fn_hook(v, k))

        for k, v in self.tensor_backup.items():
            setattr(self.tensor_module, k, self.fn_hook(v, k))

        for k, v in self.f_backup.items():
            setattr(self.f_module, k, self.fn_hook(v, k))

        for k, v in self.model_backup.items():
            setattr(self.model, k, self.fn_hook(v, k))

        return self.logs

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for k, v in self.torch_backup.items():
            setattr(self.torch_module, k, v)

        for k, v in self.tensor_backup.items():
            setattr(self.tensor_module, k, v)

        for k, v in self.f_backup.items():
            setattr(self.f_module, k, v)

        for k, v in self.model_backup.items():
            setattr(self.model, k, v)

        for handle in self.handles_pre_forward.values():
            handle.obj.remove()
        self.handles_pre_forward = {}

        for handle in self.handles_forward.values():
            handle.obj.remove()
        self.handles_forward = {}

        if self.live:
            footer = f"\nTotal params: {self.total_num_train_params + self.total_num_non_train_params:,}\n" \
                     f"Trainable params: {self.total_num_train_params:,}\n" \
                     f"Non-trainable params: {self.total_num_non_train_params:,}\n"
            print(footer)

    def get_num_params(self, module: nn.Module) -> Tuple[int, int]:
        """Counts parameters of the module (trainable, non-trainable).
        """
        try:
            # Get params
            params = module.parameters()
            # Get the first one
            param = next(params, None)
            # Module contains params
            if param is not None:
                # Count number of elements of Parameters
                num_train_params = 0
                num_non_train_params = 0
                p_id = id(param)
                if p_id not in self.param_ids:
                    self.param_ids.add(p_id)
                    if param.requires_grad:
                        num_train_params += param.numel()
                    else:
                        num_non_train_params += param.numel()
                for p in params:
                    p_id = id(p)
                    if p_id not in self.param_ids:
                        self.param_ids.add(p_id)
                        if p.requires_grad:
                            num_train_params += p.numel()
                        else:
                            num_non_train_params += p.numel()
                return num_train_params, num_non_train_params
            else:
                # If params empty
                return 0, 0
        except AttributeError:
            # No params at all
            return 0, 0

    def module_pre_forward_hook(self, module_names: List[str], module_parents: List[str], is_parent: bool) -> Callable:
        """Prepares pre forward hook.
        """

        def hook(module, inp):
            # Log here
            # self.logs.append()

            # Online summary
            if self.live:

                if not self.low_level:
                    if module._get_name() in dir(torch.nn):
                        self.fold = True

                self.depth += 1
                if self.state == 1:
                    self.curr_parent = copy(self.curr_module)
                self.state = 1

                curr_parent = ".".join(self.curr_parent)
                module_name, module_parent = adjust_module_name(module_names, module_parents, curr_parent)

                self.curr_module.append(module_name)

                full_module_name = ".".join(self.curr_module[2:])
                if full_module_name in self.fold_nodes:
                    self.force_hide = True

        return hook

    def module_forward_hook(self, module_names: List[str], module_parents: List[str], is_parent: bool) -> Callable:
        """Prepares forward hook.
        """

        def hook(module, inp, out):
            # Log here
            # self.logs.append()

            # Online summary
            if self.live:
                if not self.low_level:
                    if module._get_name() in dir(torch.nn):
                        self.fold = False

                is_comp = True if self.state == 0 else False
                self.state = 0

                self.curr_parent = self.curr_module[:-1]
                curr_parent = ".".join(self.curr_parent)
                module_name, module_parent = adjust_module_name(module_names, module_parents, curr_parent)

                if self.curr_module[-1] != module_name:
                    raise RuntimeError(f"Module name mismatch error: {self.curr_module[-1]} != {module_name}")

                # Compute size
                with self.tmp_unpatch(["size"], self.tensor_module, self.tensor_backup):
                    out_size = get_size(out)

                # Count parameters
                with self.tmp_unpatch(["dim", "unbind", "numel"], self.tensor_module, self.tensor_backup):
                    num_train_params, num_non_train_params = self.get_num_params(module)
                self.total_num_train_params += num_train_params
                self.total_num_non_train_params += num_non_train_params

                full_module_name = ".".join(self.curr_module[2:])
                display = (self.depth == 1) if self.top_level else (not is_comp or full_module_name in self.fold_nodes)

                if display:
                    line = None
                    if self.depth <= self.max_depth:
                        module_type = module._get_name()
                        if module_type.lower() not in self.hide_types:
                            if full_module_name not in self.hide_names:
                                if (not self.force_hide) or (self.force_hide and is_comp):
                                    if self.full_names:
                                        module_name = full_module_name
                                    else:
                                        module_name = self.curr_module[-1]

                                    line = f"{self.count:<{self.col1_w}}" \
                                           f"{module_name:<{self.col2_w}}" \
                                           f"{module_type:<{self.col3_w}}" \
                                           f"{size_to_str(out_size):<{self.col4_w}}" \
                                           f"{num_train_params:<{self.col5_w},}" \
                                           f"{num_non_train_params:<{self.col6_w},}"

                                    print(line)
                                    self.prev_is_suppressed = False
                                    self.count += 1
                                else:
                                    line = ""

                    if line is None:
                        if not self.exclude_hidden:
                            if not self.prev_is_suppressed:
                                line = f"{'':<{self.col1_w}}" \
                                       f"{'...':<{self.col2_w}}"
                                print(line)
                                self.prev_is_suppressed = True

                self.depth -= 1
                self.curr_module.pop()
                if full_module_name in self.fold_nodes:
                    self.force_hide = False

        return hook

    def prepare_handles(self):
        """Experimental function to handle cases when the same module
        is reused in multiple places under different names. (See Model4 from tests/conf.py.)
        """
        ids_pre_forward = set()
        ids_forward = set()

        # Iterate over modules
        for m in self.modules:
            module_id = id(m.obj)

            for type_, ids_, handles in (
                    ("_forward_pre_hooks", ids_pre_forward, self.handles_pre_forward),
                    ("_forward_hooks", ids_forward, self.handles_forward),
            ):
                if module_id not in ids_:
                    # Create a handle for a module
                    handle = RemovableHandle(getattr(m.obj, type_))
                    # Store handle in a dict
                    handles[module_id] = Handle(handle, m.obj, [m.name], [m.parent], m.is_parent)
                    # Update ids
                    ids_.add(module_id)
                else:
                    # If Module already registered append other name and parent
                    handles[module_id].names.append(m.name)
                    handles[module_id].parents.append(m.parent)

    def register_pre_forward_hooks(self):
        """Registers pre-forward hook on modules.
        """
        for handle in self.handles_pre_forward.values():
            handle.module._forward_pre_hooks[handle.obj.id] = self.module_pre_forward_hook(
                handle.names,
                handle.parents,
                handle.is_parent,
            )

    def register_forward_hooks(self):
        """Registers forward hook on modules.
        """
        for handle in self.handles_forward.values():
            handle.module._forward_hooks[handle.obj.id] = self.module_forward_hook(
                handle.names,
                handle.parents,
                handle.is_parent,
            )

    @contextmanager
    def tmp_unpatch(self, fns: List[str], module, backup):
        for fn in fns:
            setattr(module, fn, backup[fn])
        yield
        for fn in fns:
            setattr(module, fn, self.fn_hook(backup[fn], fn))

    def fn_hook(self, fn, name):

        @wraps(fn)
        def hook(*args, **kwargs):
            # Log here
            # self.logs.append()

            # Online summary
            if self.live:
                if not self.fold:
                    self.depth += 1
                    if self.state == 1:
                        self.curr_parent = copy(self.curr_module)
                    self.state = 1

                    self.curr_module.append(name)

                    full_fn_name = ".".join(self.curr_module[2:])
                    if full_fn_name in self.fold_nodes:
                        self.force_hide = True

            # Run function here
            out = fn(*args, **kwargs)

            # Log here
            # self.logs.append()

            # Online summary
            if self.live:
                if not self.fold:
                    is_comp = True if self.state == 0 else False
                    self.state = 0

                    self.curr_parent = self.curr_module[:-1]

                    if self.curr_module[-1] != name:
                        raise RuntimeError(f"Module name mismatch error: {self.curr_module[-1]} != {name}")

                    # Compute size
                    with self.tmp_unpatch(["size"], self.tensor_module, self.tensor_backup):
                        out_size = get_size(out)

                    # Count parameters
                    num_train_params, num_non_train_params = (0, 0)

                    full_fn_name = ".".join(self.curr_module[2:])
                    display = (self.depth == 1) if self.top_level else (not is_comp or full_fn_name in self.fold_nodes)

                    if display:
                        line = None
                        if self.depth <= self.max_depth:
                            fn_type = self.fn_types.get(type(fn).__name__, "unknown")
                            if fn_type.lower() not in self.hide_types:
                                if self.full_type_names:
                                    fn_type = f"{fn.__name__} ({type(fn).__name__})"
                                if full_fn_name not in self.hide_names:
                                    if (not self.force_hide) or (self.force_hide and is_comp):
                                        if self.full_names:
                                            fn_name = full_fn_name
                                        else:
                                            fn_name = self.curr_module[-1]

                                        line = f"{self.count:<{self.col1_w}}" \
                                               f"{fn_name:<{self.col2_w}}" \
                                               f"{fn_type:<{self.col3_w}}" \
                                               f"{size_to_str(out_size):<{self.col4_w}}" \
                                               f"{num_train_params:<{self.col5_w},}" \
                                               f"{num_non_train_params:<{self.col6_w},}"

                                        print(line)
                                        self.prev_is_suppressed = False
                                        self.count += 1
                                    else:
                                        line = ""
                        if line is None:
                            if not self.exclude_hidden:
                                if not self.prev_is_suppressed:
                                    line = f"{'':<{self.col1_w}}" \
                                           f"{'...':<{self.col2_w}}"
                                    print(line)
                                    self.prev_is_suppressed = True

                    self.depth -= 1
                    self.curr_module.pop()
                    if full_fn_name in self.fold_nodes:
                        self.force_hide = False

            return out

        return hook
