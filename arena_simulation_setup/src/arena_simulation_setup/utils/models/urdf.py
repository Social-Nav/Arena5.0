import asyncio
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import aiofiles
import attrs

from . import Model, ModelProvider, ModelType


class ModelProvider_URDF(ModelProvider.provides(ModelType.URDF)):

    @classmethod
    async def load(cls, model_dir, model, loader_args):

        if loader_args is None:
            loader_args = {}

        base_path = model_dir / "urdf"
        xacro_path = base_path / f"{model}.urdf.xacro"
        model_path = base_path / f"{model}.urdf"

        if not xacro_path.is_file():
            raise FileNotFoundError(f"Xacro file for model {model} not found at {xacro_path}")

        def to_string(v: Any) -> str:
            if attrs.has(type(v)):
                v = attrs.asdict(v)
            if isinstance(v, dict):
                return json.dumps(v)
            return str(v)

        cmd = [
            "ros2",
            "run",
            "xacro",
            "xacro",
            str(xacro_path),
            *(
                f"{k}:={to_string(v)}"
                for k, v
                in loader_args.items()
                if v is not None
            ),
        ]

        try:

            process = await asyncio.subprocess.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode or -1, cmd, output=stdout + stderr)
            model_desc = stdout.decode("utf-8")

            async with aiofiles.open(model_path, 'w') as f:
                await f.write(model_desc)

            base_dir = os.path.dirname(model_path)
            tree = ET.parse(model_path)
            root = tree.getroot()

            prefix = "package://jackal_description"

            # Iterate over every element in the XML tree and update 'filename' attributes
            for elem in root.iter():
                if 'filename' in elem.attrib:
                    original_path = elem.attrib['filename']
                    # Remove the specific package prefix if present
                    if original_path.startswith(prefix):
                        # Remove the prefix and any leading '/'
                        new_relative = original_path[len(prefix):].lstrip('/')
                        original_path = new_relative
                        print(f"Removed prefix: {prefix} -> New relative path: {original_path}")
                    # Convert to absolute path if it's not already
                    if not os.path.isabs(original_path):
                        abs_path = os.path.abspath(os.path.join(base_dir, original_path))
                        elem.attrib['filename'] = abs_path
                        print(f"Updated relative path to absolute: {original_path} -> {abs_path}")

            # Isaac runs inside Docker during eval and later re-opens the URDF by
            # path from that container.  Files created under the host /tmp are not
            # visible there, so keep the generated URDF in the shared robot model
            # directory instead of the process-local temp directory.
            async with aiofiles.tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".urdf",
                mode="wb",
                dir=base_dir,
            ) as tmp:
                ser = ET.tostring(root, encoding="utf-8", method="xml", xml_declaration=True)
                await tmp.write(ser)
                print(f"Converted URDF saved to temporary file: {tmp.name}")

            return Model(
                type=ModelType.URDF,
                name=model,
                description=model_desc,
                path=Path(tmp.name)
            )

        except subprocess.CalledProcessError as e:
            print(
                f"error processing model {model} URDF file {xacro_path}. refusing to load.\n{e}\n{e.output.decode('utf-8')}",
                file=sys.stderr
            )
            print(f"Command executed: {' '.join(cmd)}", file=sys.stderr)
            raise
