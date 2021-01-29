"""Model Summary.
"""
from typing import Tuple, List, Optional, Generator

from .core import HookOutput, Log, get_size, get_num_params
from .utils import size_to_str, adjust_module_name


def logger(
        live: bool = True,
        model_name: Optional[str] = None,
        hide_types: Optional[List[str]] = None,
        top_level: bool = False,
        col_widths: Tuple[int, int, int, int, int, int, int] = (5, 22, 20, 20, 12, 12, 10),
) -> Generator[List[str], Optional[HookOutput], None]:
    """A kind of logger+filter+online summary printer. Stores and processes hook outputs. Yields logs.
    """
    model_name = model_name or ""
    hide_types = hide_types or []
    col1_w, col2_w, col3_w, col4_w, col5_w, col6_w, col7_w = col_widths

    ids_ = set()
    current_module = []
    current_parent = [""]
    logs = []
    count = 1

    total_num_train_params = 0
    total_num_non_train_params = 0

    if live:
        header = f"Model: {model_name}\n\n" \
                 f"{'':<{col1_w}}Module" \
                 f"{'':<{sum(col_widths[1:3]) - 6}}{'Output Size':<{col4_w}}Params" \
                 f"{'':<{sum(col_widths[4:6]) - 6}}{'Runtime, ms':>{col7_w}}\n" \
                 f"{'':<{col1_w}}" \
                 f"{'Name':<{col2_w}}" \
                 f"{'Type':<{col3_w}}" \
                 f"{'':<{col4_w}}" \
                 f"{'Train':<{col5_w}}" \
                 f"{'Non-Train':<{col6_w}}" \
                 f"{'':>{col7_w}}"
        print(header)

    try:
        while True:
            try:
                hook_output: Optional[HookOutput] = (yield)
                if hook_output is not None:
                    if hook_output.type == "pre_forward":
                        # This handles outputs from pre-forward hooks

                        if live:
                            current_module.append(hook_output)

                            if hook_output.is_parent:
                                if len(hook_output.parents) == 1 and len(hook_output.names[0]):
                                    current_parent.append(".".join([hook_output.parents[0], hook_output.names[0]]))
                                else:
                                    # Not a perfect solution but ok (for now)
                                    current_parent.append(".".join([hook_output.parents[0], hook_output.names[0]]))

                        pre_forward = Log(
                            hook_output.type,
                            hook_output.names,
                            hook_output.parents,
                            hook_output.is_parent,
                            hook_output.module._get_name(),
                            None, None,
                            hook_output.time,
                        )

                        logs.append(pre_forward)

                    elif hook_output.type == "forward":
                        # Forward hooks

                        if live:
                            if hook_output.is_parent:
                                current_parent.pop()

                        module_type = hook_output.module._get_name()
                        out_size = get_size(hook_output.out)

                        num_params = get_num_params(hook_output.module)
                        num_train_params, num_non_train_params = num_params

                        forward = Log(
                            hook_output.type,
                            hook_output.names,
                            hook_output.parents,
                            hook_output.is_parent,
                            module_type,
                            out_size,
                            num_params,
                            hook_output.time,
                        )

                        logs.append(forward)

                        if live:
                            if top_level:
                                condition = lambda o, p: p.count(".") == 1
                            else:
                                condition = lambda o, p: not o.is_parent

                            module_name, module_parent = adjust_module_name(
                                hook_output.names,
                                hook_output.parents,
                                current_parent[-1],
                            )

                            if condition(hook_output, module_parent):
                                parent = module_parent.split(".")
                                parent = ".".join(parent[2:])
                                parent = parent + "." if parent else ""
                                full_module_name = parent + module_name

                                if module_type.lower() not in [t.lower() for t in hide_types]:
                                    if hook_output.module is current_module[-1].module:
                                        time = hook_output.time - current_module[-1].time
                                    else:
                                        time = 0.0

                                    out_size_str = size_to_str(out_size)
                                    line = f"{count:<{col1_w}}" \
                                           f"{full_module_name:<{col2_w}}" \
                                           f"{module_type:<{col3_w}}" \
                                           f"{out_size_str:<{col4_w}}" \
                                           f"{num_train_params:<{col5_w},}" \
                                           f"{num_non_train_params:<{col6_w},}" \
                                           f" {time * 10 ** 3:>{col7_w}.5f}"

                                    print(line)

                                    count += 1

                            current_module.pop()

                        # Count only basic modules
                        if not hook_output.is_parent:

                            # Count only once
                            module_id = id(hook_output.module)
                            if module_id not in ids_:
                                total_num_train_params += num_train_params
                                total_num_non_train_params += num_non_train_params
                                ids_.add(module_id)

            except StopIteration:
                if live:
                    footer = f"\nTotal params: {total_num_train_params + total_num_non_train_params:,}\n" \
                             f"Trainable params: {total_num_train_params:,}\n" \
                             f"Non-trainable params: {total_num_non_train_params:,}"
                    print(footer)

                final = Log(
                    "final",
                    None, None, None, None, None,
                    (
                        total_num_train_params + total_num_non_train_params,
                        total_num_train_params,
                        total_num_non_train_params,
                    ),
                    None,
                )
                logs.append(final)
                yield logs
    finally:
        if live:
            print()
