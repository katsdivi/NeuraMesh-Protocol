#!/usr/bin/env python3
#
# gguf_slice.py — Phase 10: Slices a GGUF model file by layer range [start, end)
# for true weight-sharded cross-device execution.
#
# Usage:
#   python3 scripts/gguf_slice.py input.gguf output.gguf start_layer end_layer
#

import sys
import re
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType

def main():
    if len(sys.argv) < 5:
        print("Usage: python3 gguf_slice.py <input.gguf> <output.gguf> <start_layer> <end_layer>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    start_layer = int(sys.argv[3])
    end_layer = int(sys.argv[4])

    if start_layer < 0 or end_layer <= start_layer:
        print(f"Invalid layer range: [{start_layer}, {end_layer})")
        sys.exit(1)

    print(f"Slicing {input_path} -> {output_path} (layers {start_layer} to {end_layer-1})...")

    if not os.path.exists(input_path):
        print(f"Input file {input_path} does not exist!")
        sys.exit(1)

    reader = GGUFReader(input_path)
    
    # 1. Determine the model architecture to identify keys
    arch = None
    for key in reader.fields.keys():
        if key == "general.architecture":
            arch = reader.fields[key].parts[-1].tobytes().decode("utf-8").strip("\x00")
            break
    
    if not arch:
        # Fallback to general lookup
        print("Warning: general.architecture metadata not found. Defaulting to 'llama'.")
        arch = "llama"

    writer = GGUFWriter(output_path, arch)

    # 2. Copy metadata key-value pairs, overriding block_count
    block_count_key = f"{arch}.block_count"
    overridden_block_count = False

    for key, field in reader.fields.items():
        # Override the block count for the sliced range
        if key == block_count_key or key.endswith(".block_count"):
            for kv in writer.kv_data:
                kv.pop(key, None)
            val = end_layer - start_layer
            writer.add_uint32(key, val)
            overridden_block_count = True
            print(f"Overriding metadata {key} to {val} (was {field.parts[-1][0]})")
        elif key == "general.name":
            for kv in writer.kv_data:
                kv.pop(key, None)
            name = field.contents()
            new_name = f"{name}_sliced_{start_layer}_{end_layer}"
            writer.add_string(key, new_name)
        else:
            if any(key in kv for kv in writer.kv_data):
                continue
            
            val = field.contents()
            
            # Map field.types array to GGUFWriter add_key_value params
            if len(field.types) > 1:
                writer.add_key_value(key, val, field.types[0], field.types[1])
            elif len(field.types) == 1:
                writer.add_key_value(key, val, field.types[0])
            else:
                writer.add_key_value(key, val, GGUFValueType.INT32)

    # 3. Determine original block count to identify if we are the last slice
    original_block_count = 0
    for key, field in reader.fields.items():
        if key == block_count_key or key.endswith(".block_count"):
            original_block_count = int(field.contents())
            break
            
    # 4. Filter and register tensor metadata
    tensors_to_copy = []
    block_pattern = re.compile(r"^blk\.(\d+)\.(.+)$")

    for tensor in reader.tensors:
        match = block_pattern.match(tensor.name)
        if match:
            layer_idx = int(match.group(1))
            suffix = match.group(2)
            if start_layer <= layer_idx < end_layer:
                new_name = f"blk.{layer_idx - start_layer}.{suffix}"
                tensors_to_copy.append((tensor, new_name))
        else:
            if tensor.name == "output_norm.weight" and end_layer < original_block_count:
                import numpy as np
                class DummyTensor:
                    def __init__(self, name, shape, data, tensor_type, endianess):
                        self.name = name
                        self.shape = shape
                        self.data = data
                        self.tensor_type = tensor_type
                        self.n_bytes = data.nbytes
                        self.endianess = endianess
                ones = np.ones(tensor.shape, dtype=np.float32)
                dummy = DummyTensor(tensor.name, tensor.shape, ones, tensor.tensor_type, reader.endianess)
                tensors_to_copy.append((dummy, tensor.name))
                print(f"Replaced output_norm.weight with all ones for intermediate slice")
            else:
                tensors_to_copy.append((tensor, tensor.name))

    print(f"Registering {len(tensors_to_copy)} / {len(reader.tensors)} tensors...")
    for tensor, new_name in tensors_to_copy:
        writer.add_tensor_info(
            new_name,
            list(tensor.shape)[::-1],
            None,
            tensor.n_bytes,
            raw_dtype=tensor.tensor_type
        )

    # 4. Write GGUF file structure
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # 5. Write actual tensor payloads
    print("Writing tensor payloads...")
    for tensor, new_name in tensors_to_copy:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)

    writer.close()
    print("Slicing complete successfully!")

if __name__ == "__main__":
    import os
    main()
