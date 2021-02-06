"""Model Summary.
"""
from copy import copy
from typing import Tuple, List, Optional, Callable

from torch.utils.hooks import RemovableHandle

from .core import Handle, get_size, get_num_params, module_walker
from .utils import size_to_str, adjust_module_name


class Summary:

    def __init__(
            self,
            model,
            live: bool = True,
            model_name: Optional[str] = None,
            full_names: bool = True,
            hide_types: Optional[List[str]] = None,
            top_level: bool = False,
            max_depth: int = 1000,
            col_widths: Tuple[int, int, int, int, int, int] = (5, 25, 25, 25, 15, 15),
    ):
        self.live = live
        self.model_name = model_name or model._get_name()
        self.full_names = full_names
        self.hide_types = hide_types or []
        self.hide_types = [t.lower() for t in self.hide_types]
        self.top_level = top_level
        self.max_depth = max_depth
        self.col_widths = col_widths
        self.col1_w, self.col2_w, self.col3_w, self.col4_w, self.col5_w, self.col6_w = self.col_widths

        self.modules = module_walker((self.model_name, model), yield_parents=True)

        self.handles_pre_forward = {}
        self.handles_forward = {}

        self.depth = -1
        self.count = 1
        self.state = 1
        self.curr_module = [""]
        self.curr_parent = [""]

        self.ids_ = set()
        self.total_num_train_params = 0
        self.total_num_non_train_params = 0

        self.logs = []

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

        return self.logs

    def __exit__(self, exc_type, exc_value, exc_traceback):
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

    def module_pre_forward_hook(self, module_names: List[str], module_parents: List[str], is_parent: bool) -> Callable:
        """Prepares pre forward hook.
        """

        def hook(module, inp):
            # Log here
            # self.logs.append()

            # Online summary
            if self.live:
                self.depth += 1
                if self.state == 1:
                    self.curr_parent = copy(self.curr_module)
                self.state = 1

                curr_parent = ".".join(self.curr_parent)
                module_name, module_parent = adjust_module_name(module_names, module_parents, curr_parent)

                if module_parent != curr_parent:
                    raise RuntimeError(f"Module parent mismatch error: {module_parent} != {curr_parent}")

                self.curr_module.append(module_name)

        return hook

    def module_forward_hook(self, module_names: List[str], module_parents: List[str], is_parent: bool) -> Callable:
        """Prepares forward hook.
        """

        def hook(module, inp, out):
            # Log here
            # self.logs.append()

            # Online summary
            if self.live:

                is_comp = True if self.state == 0 else False
                self.state = 0

                self.curr_parent = self.curr_module[:-1]
                curr_parent = ".".join(self.curr_parent)
                module_name, module_parent = adjust_module_name(module_names, module_parents, curr_parent)

                if self.curr_module[-1] != module_name:
                    raise RuntimeError(f"Module name mismatch error: {self.curr_module[-1]} != {module_name}")

                if module_parent != curr_parent:
                    raise RuntimeError(f"Module parent mismatch error: {module_parent} != {curr_parent}")

                # Compute size and parameters
                out_size = get_size(out)
                num_params = get_num_params(module)
                num_train_params, num_non_train_params = num_params

                # Count only basic modules
                if not is_comp:
                    # Count only once
                    module_id = id(module)
                    if module_id not in self.ids_:
                        self.total_num_train_params += num_train_params
                        self.total_num_non_train_params += num_non_train_params
                        self.ids_.add(module_id)

                condition = (self.depth == 1) if self.top_level else (not is_comp)

                if condition:
                    if self.depth <= self.max_depth:
                        module_type = module._get_name()
                        if module_type.lower() not in self.hide_types:
                            if self.full_names:
                                module_name = ".".join(self.curr_module[2:])
                            else:
                                module_name = self.curr_module[-1]

                            line = f"{self.count:<{self.col1_w}}" \
                                   f"{module_name:<{self.col2_w}}" \
                                   f"{module_type:<{self.col3_w}}" \
                                   f"{size_to_str(out_size):<{self.col4_w}}" \
                                   f"{num_train_params:<{self.col5_w},}" \
                                   f"{num_non_train_params:<{self.col6_w},}"

                            print(line)

                            self.count += 1

                self.depth -= 1
                self.curr_module.pop()

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
